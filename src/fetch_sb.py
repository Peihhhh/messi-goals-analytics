import argparse
import warnings
from pathlib import Path

import pandas as pd
from statsbombpy import sb

warnings.filterwarnings("ignore", message="credentials were not supplied")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MESSI_PLAYER_ID = 5503
MESSI_BIRTH_YEAR = 1987
REGULATION_MINUTES = 90

LEAGUE_CONFIGS = [
    {
        "competition_id": 11,
        "competition_name": "La Liga",
        "club": "Barcelona",
        "first_season_start_year": 2004,
        "last_season_start_year": 2020,
    },
    {
        "competition_id": 7,
        "competition_name": "Ligue 1",
        "club": "Paris Saint-Germain",
        "first_season_start_year": 2021,
        "last_season_start_year": 2022,
    },
]

DEFAULT_OUTPUT_DIR = Path("data/raw")

# Periods 1-4 are normal time and extra time; period 5 is a penalty shootout,
# which does not count as shots or goals.
LAST_PLAYED_PERIOD = 4

# Columns of shots.csv, one row per Messi shot.
SHOT_COLUMNS = [
    "match_id",
    "period",
    "minute",
    "second",
    "play_pattern",
    "x",
    "y",
    "xg",
    "outcome",
    "shot_type",
    "body_part",
    "technique",
    "first_time",
    "under_pressure",
]


# ---------------------------------------------------------------------------
# Season discovery
# ---------------------------------------------------------------------------

def get_season_start_year(season_name):
    """Return the start year from a season name like '2011/2012' -> 2011."""
    return int(str(season_name).split("/")[0])


def find_seasons_to_download():
    """Return a list of season dicts that fall within the project scope."""
    all_competitions = sb.competitions()
    seasons_to_download = []

    for league in LEAGUE_CONFIGS:
        league_seasons = all_competitions[
            all_competitions["competition_id"] == league["competition_id"]
        ]

        for _, season_row in league_seasons.iterrows():
            season_start_year = get_season_start_year(season_row["season_name"])

            if not (
                league["first_season_start_year"]
                <= season_start_year
                <= league["last_season_start_year"]
            ):
                continue

            seasons_to_download.append({
                "competition_id": league["competition_id"],
                "competition_name": league["competition_name"],
                "club": league["club"],
                "season_id": int(season_row["season_id"]),
                "season_name": season_row["season_name"],
                "season_start_year": season_start_year,
            })

    seasons_to_download.sort(
        key=lambda s: (s["season_start_year"], s["competition_id"])
    )
    return seasons_to_download


# ---------------------------------------------------------------------------
# Minutes and appearance extraction (logic from the preprocessing notebook)
# ---------------------------------------------------------------------------

def clock_string_to_minute(clock_string):
    """Convert a clock string like '67:31' to the integer minute 67."""
    return int(str(clock_string).split(":")[0])


def compute_minutes_from_positions(position_spells):
    """Estimate minutes played from StatsBomb position spells.

    Drops extra time (caps everything at 90) and ensures at least 1 minute
    for a player who came on very late in stoppage time.
    An empty list means the player did not play; returns 0.
    """
    if not position_spells:
        return 0

    first_spell = position_spells[0]
    last_spell = position_spells[-1]

    start_minute = clock_string_to_minute(first_spell["from"])
    end_minute = (
        REGULATION_MINUTES
        if last_spell["to"] is None
        else clock_string_to_minute(last_spell["to"])
    )

    # Cap both ends at 90 to drop extra time.
    start_minute = min(start_minute, REGULATION_MINUTES)
    end_minute = min(end_minute, REGULATION_MINUTES)

    # Convention: a very late substitute still counts as at least 1 minute.
    return max(end_minute - start_minute, 1)


def get_messi_appearance(match_id, club):
    """Return an appearance dict for Messi, or None if he did not play."""
    lineups_by_team = sb.lineups(match_id=match_id)

    if club not in lineups_by_team:
        return None

    club_lineup = lineups_by_team[club]
    messi_rows = club_lineup[club_lineup["player_id"] == MESSI_PLAYER_ID]

    if messi_rows.empty:
        return None

    position_spells = messi_rows.iloc[0]["positions"]
    minutes_played = compute_minutes_from_positions(position_spells)

    if minutes_played == 0:
        return None

    first_spell = position_spells[0]
    last_spell = position_spells[-1]

    return {
        "match_id": int(match_id),
        "started": int(first_spell["start_reason"] == "Starting XI"),
        "minutes_played": minutes_played,
        "starting_position": first_spell["position"],
        "end_reason": last_spell["end_reason"],
    }


# ---------------------------------------------------------------------------
# Shot extraction
# ---------------------------------------------------------------------------

def is_true(value):
    """StatsBomb leaves boolean flags empty (NaN) when they are False."""
    return bool(value == True)  # noqa: E712  (NaN == True is False)


def build_shot_row(match_id, shot_event):
    """Convert one StatsBomb shot event to the project's shots format."""
    x, y = shot_event["location"]
    return {
        "match_id": int(match_id),
        "period": int(shot_event["period"]),
        "minute": int(shot_event["minute"]),
        "second": int(shot_event["second"]),
        "play_pattern": shot_event["play_pattern"],
        "x": x,
        "y": y,
        "xg": shot_event["shot_statsbomb_xg"],
        "outcome": shot_event["shot_outcome"],
        "shot_type": shot_event["shot_type"],          # Open Play / Free Kick / Penalty
        "body_part": shot_event["shot_body_part"],
        "technique": shot_event.get("shot_technique"),
        "first_time": is_true(shot_event.get("shot_first_time")),
        "under_pressure": is_true(shot_event.get("under_pressure")),
    }


def get_messi_shots(match_id):
    """Return a list of shot dicts for Messi in one match (empty list if he did not shoot)."""
    events = sb.events(match_id=match_id)

    is_messi_shot = (
        (events["type"] == "Shot")
        & (events["player_id"] == MESSI_PLAYER_ID)
        & (events["period"] <= LAST_PLAYED_PERIOD)
    )
    messi_shots = events[is_messi_shot]

    return [build_shot_row(match_id, shot_event) for _, shot_event in messi_shots.iterrows()]


# ---------------------------------------------------------------------------
# Match row builder
# ---------------------------------------------------------------------------

def build_match_row(statsbomb_match, season):
    """Convert one StatsBomb match row to the project's matches format."""
    club = season["club"]
    is_home_match = statsbomb_match["home_team"] == club

    if is_home_match:
        opponent = statsbomb_match["away_team"]
        venue = "home"
        goals_for = statsbomb_match["home_score"]
        goals_against = statsbomb_match["away_score"]
    else:
        opponent = statsbomb_match["home_team"]
        venue = "away"
        goals_for = statsbomb_match["away_score"]
        goals_against = statsbomb_match["home_score"]

    return {
        "match_id": int(statsbomb_match["match_id"]),
        "date": str(statsbomb_match["match_date"]),
        "season": season["season_name"],
        "age": season["season_start_year"] - MESSI_BIRTH_YEAR,
        "competition": season["competition_name"],
        "match_week": int(statsbomb_match["match_week"]),
        "club": club,
        "opponent": opponent,
        "venue": venue,
        "home_score": int(statsbomb_match["home_score"]),
        "away_score": int(statsbomb_match["away_score"]),
        "goals_for": int(goals_for),
        "goals_against": int(goals_against),
    }


# ---------------------------------------------------------------------------
# Main download loop
# ---------------------------------------------------------------------------

def download_tables(seasons):
    """Fetch all matches, appearances and Messi shots from StatsBomb and return three DataFrames."""
    match_rows = []
    appearance_rows = []
    shot_rows = []

    for season in seasons:
        statsbomb_matches = sb.matches(
            competition_id=season["competition_id"],
            season_id=season["season_id"],
        )

        appearances_this_season = 0
        shots_this_season = 0

        for _, statsbomb_match in statsbomb_matches.iterrows():
            teams_in_match = (
                statsbomb_match["home_team"],
                statsbomb_match["away_team"],
            )
            if season["club"] not in teams_in_match:
                continue

            match_id = int(statsbomb_match["match_id"])
            appearance = get_messi_appearance(match_id, season["club"])

            # Skip matches where Messi was named but did not play.
            if appearance is None:
                continue

            match_rows.append(build_match_row(statsbomb_match, season))
            appearance_rows.append(appearance)
            shots_this_match = get_messi_shots(match_id)
            shot_rows.extend(shots_this_match)
            appearances_this_season += 1
            shots_this_season += len(shots_this_match)

        print(
            f"{season['competition_name']} {season['season_name']}: "
            f"{appearances_this_season} appearances, {shots_this_season} shots"
        )

    if not match_rows:
        raise ValueError(
            "No match data was produced.  "
            "Check your internet connection or the StatsBomb library."
        )

    matches_table = pd.DataFrame(match_rows).sort_values("date").reset_index(drop=True)

    # Align appearances to match order so match_id sequences are identical.
    appearances_table = matches_table[["match_id"]].merge(
        pd.DataFrame(appearance_rows), on="match_id", how="left", validate="one_to_one"
    )

    # columns= keeps the header even if no shot was found.
    shots_table = pd.DataFrame(shot_rows, columns=SHOT_COLUMNS)

    return matches_table, appearances_table, shots_table


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_sanity_checks(matches_table, appearances_table, shots_table):
    """Print a quick structural check; raises AssertionError on failure."""
    assert matches_table["match_id"].is_unique, "Duplicate match_id in matches"
    assert appearances_table["match_id"].is_unique, "Duplicate match_id in appearances"
    assert len(matches_table) == len(appearances_table), "Row count mismatch"
    assert matches_table["match_id"].equals(appearances_table["match_id"]), (
        "match_id order or set differs between the two tables"
    )
    assert appearances_table[["started", "minutes_played"]].notna().all().all()
    assert appearances_table["started"].isin([0, 1]).all()
    assert appearances_table["minutes_played"].between(1, 90).all()
    assert shots_table["match_id"].isin(matches_table["match_id"]).all(), (
        "Shot found for a match that is not in matches"
    )
    assert shots_table["xg"].between(0, 1).all(), "xG missing or outside 0-1"

    print(f"Matches: {len(matches_table)}")
    print(f"Starts: {int(appearances_table['started'].sum())}")
    print(f"Total estimated minutes: {int(appearances_table['minutes_played'].sum())}")
    print(f"Shots: {len(shots_table)}, goals from shots: {int((shots_table['outcome'] == 'Goal').sum())}")
    print("All structural checks passed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Download StatsBomb open data for Messi's league appearances."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Folder to write appearances.csv, matches.csv and shots.csv (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    print("Finding seasons to download ...")
    seasons = find_seasons_to_download()
    print(f"Found {len(seasons)} seasons across {len(LEAGUE_CONFIGS)} leagues.\n")

    print("Downloading match, lineup and event data from StatsBomb ...")
    matches_table, appearances_table, shots_table = download_tables(seasons)

    print("\nRunning sanity checks ...")
    run_sanity_checks(matches_table, appearances_table, shots_table)

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    matches_path = arguments.output_dir / "matches.csv"
    appearances_path = arguments.output_dir / "appearances.csv"
    shots_path = arguments.output_dir / "shots.csv"

    matches_table.to_csv(matches_path, index=False)
    appearances_table.to_csv(appearances_path, index=False)
    shots_table.to_csv(shots_path, index=False)

    print(f"\nSaved: {matches_path}")
    print(f"Saved: {appearances_path}")
    print(f"Saved: {shots_path}")
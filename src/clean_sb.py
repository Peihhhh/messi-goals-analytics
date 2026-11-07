import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_APPEARANCES_PATH = Path("data/raw/appearances.csv")
DEFAULT_MATCHES_PATH = Path("data/raw/matches.csv")
DEFAULT_SHOTS_PATH = Path("data/raw/shots.csv")
DEFAULT_MATCHES_OUTPUT_PATH = Path("data/clean/messi_league_matches.csv")
DEFAULT_SHOTS_OUTPUT_PATH = Path("data/clean/messi_league_shots.csv")

# StatsBomb names -> names used in the cleaned Kaggle file, so labels match in Power BI.
COMPETITION_NAME_MAP = {"La Liga": "LaLiga", "Ligue 1": "Ligue 1"}
CLUB_NAME_MAP = {"Barcelona": "FC Barcelona", "Paris Saint-Germain": "Paris Saint-Germain"}

# StatsBomb starting positions grouped into a few roles.
POSITION_GROUPS = {
    "Right Wing": "Wing",
    "Left Wing": "Wing",
    "Center Forward": "Forward",
    "Right Center Forward": "Forward",
    "Left Center Forward": "Forward",
    "Center Attacking Midfield": "Attacking midfield",
    "Right Attacking Midfield": "Attacking midfield",
}
OTHER_POSITION_GROUP = "Other"

# StatsBomb shot outcomes grouped for charts.
# "Post" and "Saved Off Target" were going wide, so they count as off target.
SHOT_OUTCOME_GROUPS = {
    "Goal": "Goal",
    "Saved": "Saved",
    "Saved to Post": "Saved",
    "Off T": "Off target",
    "Post": "Off target",
    "Wayward": "Off target",
    "Saved Off Target": "Off target",
    "Blocked": "Blocked",
}
ON_TARGET_GROUPS = {"Goal", "Saved"}

# StatsBomb pitch is 120 x 80; the goal is at x = 120, posts at y = 36 and y = 44.
GOAL_X = 120.0
GOAL_CENTER_Y = 40.0
GOAL_WIDTH = 8.0

MATCH_COLUMNS = [
    "match_id", "date", "season", "age", "club", "competition", "matchday",
    "opponent", "venue", "started", "starting_position", "position_group",
    "minutes_played", "end_reason", "team_goals", "opponent_goals", "match_result",
    "shots", "shots_on_target", "non_penalty_shots", "xg", "npxg",
    "goals", "non_penalty_goals", "team_goals_without_messi",
]

SHOT_COLUMNS = [
    "shot_id", "match_id", "date", "season", "age", "competition", "club",
    "period", "minute", "second", "play_pattern", "x", "y",
    "distance_to_goal", "shot_angle_deg", "xg", "is_penalty", "outcome",
    "outcome_group", "is_goal", "is_on_target", "shot_type", "body_part",
    "technique", "first_time", "under_pressure",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_statsbomb_files(appearances_path, matches_path):
    """Read both raw files and join them on match_id (exactly one row each)."""
    appearances = pd.read_csv(appearances_path)
    matches = pd.read_csv(matches_path, parse_dates=["date"])

    # validate="one_to_one" raises an error if a match_id appears twice in either file.
    merged = matches.merge(appearances, on="match_id", how="inner", validate="one_to_one")
    assert len(merged) == len(matches) == len(appearances), "matches and appearances differ"
    return merged


# ---------------------------------------------------------------------------
# Match table
# ---------------------------------------------------------------------------

def standardize_names(dataframe):
    """Use the same competition, club, season and venue labels as the Kaggle file."""
    dataframe["competition"] = dataframe["competition"].map(COMPETITION_NAME_MAP)
    dataframe["club"] = dataframe["club"].map(CLUB_NAME_MAP)

    # '2004/2005' -> '2004-05'
    dataframe["season"] = dataframe["season"].str.slice(0, 4) + "-" + dataframe["season"].str.slice(7, 9)

    dataframe["venue"] = dataframe["venue"].str.capitalize()  # 'home' -> 'Home'
    return dataframe.rename(columns={"match_week": "matchday"})


def add_team_score_columns(dataframe):
    """Team and opponent goals from Messi's team point of view, plus W/D/L."""
    dataframe = dataframe.rename(columns={"goals_for": "team_goals", "goals_against": "opponent_goals"})
    dataframe["match_result"] = np.select(
        [dataframe["team_goals"] > dataframe["opponent_goals"],
         dataframe["team_goals"] == dataframe["opponent_goals"]],
        ["W", "D"],
        default="L",
    )
    return dataframe.drop(columns=["home_score", "away_score"])


def add_position_columns(dataframe):
    """
    Group the starting position.

    For a substitute, StatsBomb records the position of the player he replaced
    (this is why Messi appears as 'Right Back' twice), so only trust
    position_group when started is True.
    """
    dataframe["started"] = dataframe["started"].astype(bool)
    dataframe["position_group"] = (
        dataframe["starting_position"].map(POSITION_GROUPS).fillna(OTHER_POSITION_GROUP)
    )
    return dataframe


def summarize_shots_per_match(shots):
    """Turn the shot table (one row per shot) into one row per match."""
    shots = shots.assign(
        non_penalty_shot=~shots["is_penalty"],
        non_penalty_xg=shots["xg"].where(~shots["is_penalty"], 0),
        non_penalty_goal=shots["is_goal"] & ~shots["is_penalty"],
    )
    return (
        shots.groupby("match_id")
        .agg(
            shots=("xg", "size"),
            shots_on_target=("is_on_target", "sum"),
            non_penalty_shots=("non_penalty_shot", "sum"),
            xg=("xg", "sum"),
            npxg=("non_penalty_xg", "sum"),
            goals=("is_goal", "sum"),
            non_penalty_goals=("non_penalty_goal", "sum"),
        )
        .reset_index()
    )


def add_shot_columns(dataframe, shots):
    """Add Messi's shots, xG and goals per match. No shot row means he did not shoot -> 0."""
    shots_per_match = summarize_shots_per_match(shots)
    shot_columns = [column for column in shots_per_match.columns if column != "match_id"]

    dataframe = dataframe.merge(shots_per_match, on="match_id", how="left", validate="one_to_one")
    dataframe[shot_columns] = dataframe[shot_columns].fillna(0)

    count_columns = [column for column in shot_columns if column not in ("xg", "npxg")]
    dataframe[count_columns] = dataframe[count_columns].astype(int)

    dataframe["team_goals_without_messi"] = dataframe["team_goals"] - dataframe["goals"]
    return dataframe


def build_match_table(appearances_path, matches_path, shots):
    """Run every match-level step in order and return the cleaned match table."""
    dataframe = load_statsbomb_files(appearances_path, matches_path)
    dataframe = standardize_names(dataframe)
    dataframe = add_team_score_columns(dataframe)
    dataframe = add_position_columns(dataframe)
    dataframe = add_shot_columns(dataframe, shots)
    dataframe = dataframe.sort_values("date", kind="stable").reset_index(drop=True)
    return dataframe[MATCH_COLUMNS]


# ---------------------------------------------------------------------------
# Shot table
# ---------------------------------------------------------------------------

def compute_distance_to_goal(x, y):
    """Distance from the shot location to the centre of the goal (StatsBomb units, ~yards)."""
    return np.hypot(GOAL_X - x, GOAL_CENTER_Y - y)


def compute_shot_angle(x, y):
    """
    Angle (degrees) between the two posts as seen from the shot location.
    Wide angle = lots of goal to aim at; used as a feature in the xG model.
    """
    dx = GOAL_X - x
    dy = GOAL_CENTER_Y - y
    half_width = GOAL_WIDTH / 2
    angle = np.arctan2(GOAL_WIDTH * dx, dx**2 + dy**2 - half_width**2)
    return np.degrees(angle)


def add_shot_features(shots):
    """Flags and geometry used by both the dashboard and the xG model."""
    shots = shots.copy()
    shots["is_penalty"] = shots["shot_type"] == "Penalty"
    shots["outcome_group"] = shots["outcome"].map(SHOT_OUTCOME_GROUPS)
    shots["is_goal"] = shots["outcome"] == "Goal"
    shots["is_on_target"] = shots["outcome_group"].isin(ON_TARGET_GROUPS)
    shots["distance_to_goal"] = compute_distance_to_goal(shots["x"], shots["y"]).round(2)
    shots["shot_angle_deg"] = compute_shot_angle(shots["x"], shots["y"]).round(2)
    return shots


def load_shots(shots_path):
    """Read shots.csv and add the derived shot features."""
    shots = pd.read_csv(shots_path)
    return add_shot_features(shots)


def build_shot_table(shots, match_table):
    """One row per shot, with match context so Power BI can filter shots directly."""
    match_context = match_table[["match_id", "date", "season", "age", "competition", "club"]]
    shot_table = shots.merge(match_context, on="match_id", how="left", validate="many_to_one")
    shot_table = shot_table.sort_values(["date", "period", "minute", "second"], kind="stable")
    shot_table["shot_id"] = np.arange(1, len(shot_table) + 1)
    return shot_table[SHOT_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def check_goals_against_kaggle(match_table, goals_path):
    """
    Optional cross-check against an independent source (cleaned Kaggle goals file).

    StatsBomb goals per match must equal the Kaggle goals per match. Also report
    Kaggle league goals whose match is missing from StatsBomb. The analysis itself
    uses StatsBomb only, so this check is skipped when no Kaggle file is given.
    Last result: 0 mismatches in 577 matches; StatsBomb lacks 2007-09-26
    Barcelona 4-1 Real Zaragoza (2 Messi goals).
    """
    if goals_path is None:
        print("No Kaggle goals file given: cross-check skipped.")
        return
    goals = pd.read_csv(goals_path)
    league_goals = goals[goals["competition"].isin(COMPETITION_NAME_MAP.values())]

    missing = league_goals[league_goals["match_id"].isna()]
    if len(missing) > 0:
        print(f"Kaggle league goals with no StatsBomb match (left out of the analysis): {len(missing)}")
        print(missing[["date", "competition", "opponent", "goal_minute"]].to_string(index=False))

    kaggle_goals_per_match = (
        league_goals.dropna(subset=["match_id"])
        .astype({"match_id": int})
        .groupby("match_id")
        .size()
        .rename("kaggle_goals")
    )
    compared = match_table.set_index("match_id").join(kaggle_goals_per_match).fillna({"kaggle_goals": 0})
    mismatched = compared[compared["goals"] != compared["kaggle_goals"]]
    assert mismatched.empty, f"StatsBomb and Kaggle goals differ:\n{mismatched[['date', 'goals', 'kaggle_goals']]}"


def run_sanity_checks(match_table, shot_table):
    """Stop the script if anything that must hold does not hold."""
    assert match_table["match_id"].is_unique, "Duplicate match_id"
    assert match_table[["competition", "club"]].notna().all().all(), "Unmapped competition or club"
    assert match_table["minutes_played"].between(1, 90).all(), "Minutes outside 1-90"
    assert (match_table["goals"] <= match_table["team_goals"]).all(), "Messi goals > team goals"
    assert not (~match_table["started"] & (match_table["minutes_played"] >= 90)).any(), (
        "Substitute with 90 minutes: re-run fetch_sb.py (started is now read from the kick-off time)"
    )
    assert shot_table["outcome_group"].notna().all(), "Unknown shot outcome, update SHOT_OUTCOME_GROUPS"
    assert shot_table["competition"].notna().all(), "Shot without a match"
    assert shot_table["is_goal"].sum() == match_table["goals"].sum(), "Shot and match goal totals differ"

    print(f"Matches: {len(match_table)}, minutes: {match_table['minutes_played'].sum()}")
    print(f"Shots: {len(shot_table)}, goals: {match_table['goals'].sum()}")
    print("All checks passed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description="Build the cleaned StatsBomb match and shot tables.")
    parser.add_argument("--appearances", type=Path, default=DEFAULT_APPEARANCES_PATH)
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES_PATH)
    parser.add_argument(
        "--goals",
        type=Path,
        default=None,
        help="Optional cleaned Kaggle goals file, only used to cross-check goal counts",
    )
    parser.add_argument("--shots", type=Path, default=DEFAULT_SHOTS_PATH)
    parser.add_argument("--matches-output", type=Path, default=DEFAULT_MATCHES_OUTPUT_PATH)
    parser.add_argument("--shots-output", type=Path, default=DEFAULT_SHOTS_OUTPUT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    shots = load_shots(arguments.shots)
    match_table = build_match_table(arguments.appearances, arguments.matches, shots)
    shot_table = build_shot_table(shots, match_table)

    check_goals_against_kaggle(match_table, arguments.goals)
    run_sanity_checks(match_table, shot_table)

    arguments.matches_output.parent.mkdir(parents=True, exist_ok=True)
    match_table.to_csv(arguments.matches_output, index=False)
    shot_table.to_csv(arguments.shots_output, index=False)
    print(f"Saved: {arguments.matches_output}")
    print(f"Saved: {arguments.shots_output}")
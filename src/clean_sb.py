import argparse
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_APPEARANCES_PATH = Path("data/raw/appearances.csv")
DEFAULT_MATCHES_PATH = Path("data/raw/matches.csv")
DEFAULT_GOALS_PATH = Path("data/clean/messi_goals_clean.csv")
DEFAULT_SHOTS_PATH = Path("data/raw/shots.csv")
DEFAULT_OUTPUT_PATH = Path("data/clean/messi_league_matches.csv")

# Matches added to the raw files by hand because they are missing from StatsBomb
# open data. They have no event data, so their shot columns stay empty (NaN)
# instead of 0, which would wrongly mean "Messi did not shoot".
MATCHES_WITHOUT_EVENT_DATA = {99999}

SHOT_OUTCOMES_ON_TARGET = {"Goal", "Saved", "Saved to Post"}

# StatsBomb names -> names used in the cleaned Kaggle file, so the two can be joined.
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

# What each output column may be used for in the ML part.
#   target       : what a model predicts
#   pre_match    : known before kick-off, safe to use as a feature
#   post_match   : only known after the match, must NOT be used as a feature
#   id_or_quality: identifiers and data-quality flags, not features
COLUMN_ROLES = {
    "match_id": "id_or_quality",
    "date": "id_or_quality",
    "season": "pre_match",
    "age": "pre_match",
    "club": "pre_match",
    "competition": "pre_match",
    "matchday": "pre_match",
    "opponent": "pre_match",
    "venue": "pre_match",
    "started": "pre_match",
    "starting_position": "pre_match",
    "position_group": "pre_match",
    "minutes_played": "post_match",
    "end_reason": "post_match",
    "team_goals": "post_match",
    "opponent_goals": "post_match",
    "match_result": "post_match",
    "team_goals_without_messi": "post_match",
    # Shot columns describe what happened in the match, so they are post_match:
    # using them as features to predict messi_goals would leak the answer.
    "shots": "post_match",
    "shots_on_target": "post_match",
    "non_penalty_shots": "post_match",
    "xg": "post_match",
    "npxg": "post_match",
    "goals_from_shots": "post_match",
    "non_penalty_goals": "post_match",
    "messi_goals": "target",
    "scored": "target",
    "position_reliable": "id_or_quality",
    "flag_sub_with_90_minutes": "id_or_quality",
    "shot_data_available": "id_or_quality",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_statsbomb_files(appearances_path, matches_path):
    """Read both raw files and join them on match_id (exactly one row each)."""
    appearances = pd.read_csv(appearances_path)
    matches = pd.read_csv(matches_path, parse_dates=["date"])

    # validate="one_to_one" raises an error if a match_id appears twice in either file.
    merged = matches.merge(appearances, on="match_id", how="inner", validate="one_to_one")

    lost_rows = max(len(matches), len(appearances)) - len(merged)
    print(f"Matches: {len(matches)}, appearances: {len(appearances)}, joined: {len(merged)} (lost {lost_rows})")
    return merged


# ---------------------------------------------------------------------------
# Cleaning steps
# ---------------------------------------------------------------------------

def standardize_names(dataframe):
    """Use the same competition, club, season and venue labels as the Kaggle file."""
    dataframe["competition"] = dataframe["competition"].map(COMPETITION_NAME_MAP)
    dataframe["club"] = dataframe["club"].map(CLUB_NAME_MAP)

    # '2004/2005' -> '2004-05'
    first_year = dataframe["season"].str.slice(0, 4)
    second_year_short = dataframe["season"].str.slice(7, 9)
    dataframe["season"] = first_year + "-" + second_year_short

    dataframe["venue"] = dataframe["venue"].str.capitalize()  # 'home' -> 'Home'
    dataframe = dataframe.rename(columns={"match_week": "matchday"})
    return dataframe


def add_team_score_columns(dataframe):
    """Team and opponent goals from Messi's team point of view, plus W/D/L."""
    dataframe = dataframe.rename(
        columns={"goals_for": "team_goals", "goals_against": "opponent_goals"}
    )

    def label_result(row):
        if row["team_goals"] > row["opponent_goals"]:
            return "W"
        if row["team_goals"] == row["opponent_goals"]:
            return "D"
        return "L"

    dataframe["match_result"] = dataframe.apply(label_result, axis=1)
    dataframe = dataframe.drop(columns=["home_score", "away_score"])
    return dataframe


def add_position_columns(dataframe):
    """
    Group the starting position and mark when it can be trusted.

    For a substitute, StatsBomb records the position of the player he replaced
    (this is why Messi appears as 'Right Back' twice), so the position is only
    reliable when he started the match.
    """
    dataframe["position_group"] = (
        dataframe["starting_position"].map(POSITION_GROUPS).fillna(OTHER_POSITION_GROUP)
    )
    dataframe["started"] = dataframe["started"].astype(bool)
    full_match_mask = (~dataframe["started"]) & (dataframe["minutes_played"] >= 90)
    dataframe.loc[full_match_mask, "started"] = True
    dataframe["position_reliable"] = dataframe["started"]
    return dataframe


def add_quality_flags(dataframe):
    """Flag rows that look inconsistent so analyses can decide what to do with them."""
    dataframe["flag_sub_with_90_minutes"] = (~dataframe["started"]) & (
        dataframe["minutes_played"] >= 90
    )
    return dataframe


def add_messi_goals(dataframe, goals_path):
    """
    Count Messi's goals per match from the cleaned Kaggle file and add them.

    The two sources spell opponents differently, so the join key is
    (date, competition). Matches where he did not score get 0.
    """
    goals = pd.read_csv(goals_path, parse_dates=["date"])
    league_goals = goals[goals["competition"].isin(COMPETITION_NAME_MAP.values())]

    goals_per_match = (
        league_goals.groupby(["date", "competition"]).size().rename("messi_goals").reset_index()
    )

    dataframe = dataframe.merge(goals_per_match, on=["date", "competition"], how="left")
    dataframe["messi_goals"] = dataframe["messi_goals"].fillna(0).astype(int)
    dataframe["scored"] = dataframe["messi_goals"] > 0
    dataframe["team_goals_without_messi"] = dataframe["team_goals"] - dataframe["messi_goals"]

    # Report Kaggle league matches that have no StatsBomb row.
    statsbomb_keys = set(zip(dataframe["date"], dataframe["competition"]))
    is_unmatched = [
        (match_date, competition) not in statsbomb_keys
        for match_date, competition in zip(goals_per_match["date"], goals_per_match["competition"])
    ]
    unmatched = goals_per_match[is_unmatched]
    print(
        f"League goals in the Kaggle file: {len(league_goals)}, "
        f"matched to a StatsBomb match: {int(dataframe['messi_goals'].sum())}"
    )
    if len(unmatched) > 0:
        print("Kaggle league matches with no StatsBomb row:")
        print(unmatched.to_string(index=False))
    return dataframe


def summarize_shots_per_match(shots):
    """Turn shots.csv (one row per shot) into one row per match."""
    is_penalty = shots["shot_type"] == "Penalty"
    is_goal = shots["outcome"] == "Goal"

    shots = shots.assign(
        on_target=shots["outcome"].isin(SHOT_OUTCOMES_ON_TARGET),
        non_penalty_shot=~is_penalty,
        non_penalty_xg=shots["xg"].where(~is_penalty, 0),
        goal=is_goal,
        non_penalty_goal=is_goal & ~is_penalty,
    )
    return (
        shots.groupby("match_id")
        .agg(
            shots=("xg", "size"),
            shots_on_target=("on_target", "sum"),
            non_penalty_shots=("non_penalty_shot", "sum"),
            xg=("xg", "sum"),
            npxg=("non_penalty_xg", "sum"),
            goals_from_shots=("goal", "sum"),
            non_penalty_goals=("non_penalty_goal", "sum"),
        )
        .reset_index()
    )


def add_shot_columns(dataframe, shots_path):
    """
    Add Messi's shots and xG per match from shots.csv.

    A match with no row in shots.csv means he played but did not shoot, so it
    gets 0. Matches in MATCHES_WITHOUT_EVENT_DATA have no event data and stay NaN.
    """
    shots = pd.read_csv(shots_path)
    shots_per_match = summarize_shots_per_match(shots)
    shot_columns = [column for column in shots_per_match.columns if column != "match_id"]

    dataframe = dataframe.merge(shots_per_match, on="match_id", how="left", validate="one_to_one")

    dataframe["shot_data_available"] = ~dataframe["match_id"].isin(MATCHES_WITHOUT_EVENT_DATA)
    has_data = dataframe["shot_data_available"]
    dataframe.loc[has_data, shot_columns] = dataframe.loc[has_data, shot_columns].fillna(0)

    count_columns = [column for column in shot_columns if column not in ("xg", "npxg")]
    dataframe[count_columns] = dataframe[count_columns].astype("Int64")  # integer that allows NaN

    print(
        f"Shots in shots.csv: {len(shots)}, "
        f"matches without shot data: {int((~has_data).sum())}"
    )
    return dataframe


def reorder_columns(dataframe):
    """Put the columns in the order listed in COLUMN_ROLES, sorted by date."""
    dataframe = dataframe.sort_values("date", kind="stable").reset_index(drop=True)
    return dataframe[list(COLUMN_ROLES.keys())]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_sanity_checks(dataframe):
    """Print checks that should all pass."""
    print(f"Rows: {len(dataframe)}")
    print(f"Duplicate match_id: {int(dataframe['match_id'].duplicated().sum())}")
    print(f"Unmapped competition or club: {int(dataframe[['competition', 'club']].isna().sum().sum())}")

    too_many_goals = dataframe["messi_goals"] > dataframe["team_goals"]
    print(f"Matches where Messi goals > team goals: {int(too_many_goals.sum())}")

    bad_minutes = ~dataframe["minutes_played"].between(1, 90)
    print(f"Rows with minutes outside 1-90: {int(bad_minutes.sum())}")

    print(f"Rows flagged as substitute with 90 minutes: {int(dataframe['flag_sub_with_90_minutes'].sum())}")

    # Goals counted from StatsBomb shots should equal goals from the Kaggle file.
    has_shot_data = dataframe["shot_data_available"]
    goal_mismatch = has_shot_data & (dataframe["goals_from_shots"] != dataframe["messi_goals"])
    print(f"Matches where StatsBomb goals != Kaggle goals: {int(goal_mismatch.sum())}")
    if goal_mismatch.any():
        print(dataframe.loc[goal_mismatch, ["match_id", "date", "messi_goals", "goals_from_shots"]].to_string(index=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_league_matches(appearances_path, matches_path, goals_path, shots_path):
    """Run every step in order and return the cleaned match table."""
    dataframe = load_statsbomb_files(appearances_path, matches_path)
    dataframe = standardize_names(dataframe)
    dataframe = add_team_score_columns(dataframe)
    dataframe = add_position_columns(dataframe)
    dataframe = add_quality_flags(dataframe)
    dataframe = add_messi_goals(dataframe, goals_path)
    dataframe = add_shot_columns(dataframe, shots_path)
    dataframe = reorder_columns(dataframe)
    return dataframe


def parse_arguments():
    parser = argparse.ArgumentParser(description="Build the StatsBomb league match table.")
    parser.add_argument("--appearances", type=Path, default=DEFAULT_APPEARANCES_PATH)
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES_PATH)
    parser.add_argument("--goals", type=Path, default=DEFAULT_GOALS_PATH)
    parser.add_argument("--shots", type=Path, default=DEFAULT_SHOTS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    league_matches = build_league_matches(
        arguments.appearances, arguments.matches, arguments.goals, arguments.shots
    )
    run_sanity_checks(league_matches)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    league_matches.to_csv(arguments.output, index=False)
    print(f"Saved league match table to {arguments.output}")
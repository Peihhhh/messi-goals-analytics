import argparse
import re
from pathlib import Path

import pandas as pd

# CONSTANTS

DEFAULT_INPUT_PATH = Path("data/raw/messi_all_goals.csv")
DEFAULT_STATSBOMB_MATCHES_PATH = Path("data/raw/matches.csv")
DEFAULT_OUTPUT_PATH = Path("data/clean/messi_goals_clean.csv")

# StatsBomb competition names -> names used in the Kaggle file.
STATSBOMB_COMPETITION_MAP = {"La Liga": "LaLiga", "Ligue 1": "Ligue 1"}

EXCLUDED_CLUBS = ["FC Barcelona B"]
# Value used in the raw file when a goal has no assist (penalty, solo run, ...).
NO_ASSIST_LABEL = "Not Applicable"
 
# Value used in the raw file when the goal type is genuinely missing.
UNKNOWN_GOAL_TYPE_LABEL = "Not reported"
 
# Mapping from the 13 raw goal types to a smaller set of groups.
GOAL_TYPE_GROUPS = {
    "Left-footed shot": "Left foot",
    "Right-footed shot": "Right foot",
    "Header": "Header",
    "Penalty": "Penalty",
    "Direct free kick": "Free kick",
    "Tap-in": "Other",
    "Solo run": "Other",
    "Penalty rebound": "Other",
    "Deflected shot on goal": "Other",
    "Chest": "Other",
    "Counter attack goal": "Other",
    "Long distance kick": "Other",
    UNKNOWN_GOAL_TYPE_LABEL: "Unknown",
}

# Order used when plotting the minute buckets.
MINUTE_BUCKET_ORDER = ["0-15", "16-30", "31-45", "46-60", "61-75", "76-90", "91+"]

# HELPER FUNCTIONS
def remove_excluded_clubs(dataframe):
    """Drop goals scored for the clubs listed in EXCLUDED_CLUBS (see above)."""
    is_excluded = dataframe["club"].isin(EXCLUDED_CLUBS)
    print(f"Removed {int(is_excluded.sum())} goals from excluded clubs: {EXCLUDED_CLUBS}")
    return dataframe.loc[~is_excluded].reset_index(drop=True)

def parse_score_text(score_text):
    """
    Turn a raw score string such as '1:02' into the tuple (1, 2).
 
    In the raw file the format is 'home:away' and the away part is zero padded
    (it looks like a spreadsheet read the score as a time, h:mm). So '1:02'
    means home 1, away 2, and '10:01' would mean home 10, away 1.
    """
    parts = str(score_text).split(":")
    if len(parts) != 2:
        raise ValueError(f"Cannot parse score text: {score_text!r}")
    home_goals = int(parts[0])
    away_goals = int(parts[1])
    
    return home_goals, away_goals

def split_match_stage(stage_value):
    """
    The raw 'match_stage' column mixes two different things:
      - a round name for cup games ('Group Stage', 'Final', 'Round of 16', ...)
      - a league matchday number as text ('1', '25', '38', ...)
 
    Return (stage_name, matchday). Matchday is None for cup rounds, and
    stage_name is 'League' for numbered matchdays.
    """
    stage_text = str(stage_value).strip()
    if stage_text.isdigit():
        return "League", int(stage_text)
    return stage_text, None

def group_goal_type(raw_goal_type):
    """Map a raw goal type to one of the smaller groups defined above."""
    return GOAL_TYPE_GROUPS.get(raw_goal_type, "Other")

# MAIN CLEANING STEPS

def load_raw_data(input_path):
    """Read the raw CSV with every column as text so nothing is guessed wrongly."""
    # utf-8-sig removes the invisible byte order mark some Kaggle files start with.
    raw_dataframe = pd.read_csv(input_path, encoding="utf-8-sig", dtype=str)
    raw_dataframe.columns = [column.strip() for column in raw_dataframe.columns]
    return raw_dataframe

def clean_dates(dataframe):
    """Convert 'date' from text like '9/5/2004' (month/day/year) to a real date."""
    dataframe["date"] = pd.to_datetime(dataframe["date"], format="%m/%d/%Y")
    dataframe["year"] = dataframe["date"].dt.year
    dataframe["month"] = dataframe["date"].dt.month
    return dataframe
 
 
def clean_stage(dataframe):
    """Split 'match_stage' into 'stage_name' and 'matchday'."""
    stage_pairs = dataframe["match_stage"].apply(split_match_stage)
    dataframe["stage_name"] = stage_pairs.apply(lambda pair: pair[0])
    dataframe["matchday"] = stage_pairs.apply(lambda pair: pair[1]).astype("Int64")
    dataframe = dataframe.drop(columns=["match_stage"])
    return dataframe
 
 
def clean_scores(dataframe):
    """
    Build numeric score columns.
 
    Raw 'match_score' is the final score (home:away).
    Raw 'score_at_goal' is the score right AFTER Messi's goal (home:away).
 
    New columns, all from Messi's team point of view:
      team_goals_final, opponent_goals_final, match_result (W/D/L),
      team_goals_after, opponent_goals_after,
      team_goals_before, opponent_goals_before,
      score_diff_before (negative means the team was losing before the goal),
      game_state_before ('Losing', 'Drawing', 'Winning').
    """
    final_scores = dataframe["match_score"].apply(parse_score_text)
    after_scores = dataframe["score_at_goal"].apply(parse_score_text)
 
    home_final = final_scores.apply(lambda pair: pair[0])
    away_final = final_scores.apply(lambda pair: pair[1])
    home_after = after_scores.apply(lambda pair: pair[0])
    away_after = after_scores.apply(lambda pair: pair[1])
 
    is_home_game = dataframe["venue"] == "Home"
 
    # Pick the correct side depending on whether Messi's team played at home.
    dataframe["team_goals_final"] = home_final.where(is_home_game, away_final)
    dataframe["opponent_goals_final"] = away_final.where(is_home_game, home_final)
    dataframe["team_goals_after"] = home_after.where(is_home_game, away_after)
    dataframe["opponent_goals_after"] = away_after.where(is_home_game, home_after)
 
    # The goal itself adds one to the team's tally, so remove it for "before".
    dataframe["team_goals_before"] = dataframe["team_goals_after"] - 1
    dataframe["opponent_goals_before"] = dataframe["opponent_goals_after"]
    dataframe["score_diff_before"] = (
        dataframe["team_goals_before"] - dataframe["opponent_goals_before"]
    )
 
    def label_game_state(score_diff):
        if score_diff < 0:
            return "Losing"
        if score_diff == 0:
            return "Drawing"
        return "Winning"
 
    dataframe["game_state_before"] = dataframe["score_diff_before"].apply(label_game_state)
 
    def label_result(row):
        if row["team_goals_final"] > row["opponent_goals_final"]:
            return "W"
        if row["team_goals_final"] == row["opponent_goals_final"]:
            return "D"
        return "L"
 
    dataframe["match_result"] = dataframe.apply(label_result, axis=1)
 
    dataframe = dataframe.drop(columns=["match_score", "score_at_goal"])
    return dataframe
 
 
def clean_assists(dataframe):
    """
    'Not Applicable' means the goal had no assist (penalty, solo run, ...).
    That is different from missing data, so keep it as a clear flag.
    """
    dataframe["has_assist"] = dataframe["assist_player"] != NO_ASSIST_LABEL
    dataframe["assist_player"] = dataframe["assist_player"].where(
        dataframe["has_assist"], other=pd.NA
    )
    return dataframe
 
 
def clean_goal_types(dataframe):
    """Keep the raw goal type and add a grouped version for easier charts."""
    dataframe["goal_type_raw"] = dataframe["goal_type"]
    dataframe["goal_type"] = dataframe["goal_type_raw"].apply(group_goal_type)
    return dataframe
 
 
def clean_flags_and_categories(dataframe):
    """Convert text booleans to bool, fix dtypes, set the minute bucket order."""
    dataframe["is_home_goal"] = dataframe["is_home_goal"].str.upper() == "TRUE"
    dataframe["goal_minute"] = dataframe["goal_minute"].astype(int)
    dataframe["goal_minute_bucket"] = pd.Categorical(
        dataframe["goal_minute_bucket"], categories=MINUTE_BUCKET_ORDER, ordered=True
    )
    return dataframe
 
 
def add_goal_match_id(dataframe):
    """
    Create 'goal_match_id' (one id per match in THIS file, all competitions)
    and 'goal_number_in_match'. A match is identified by its date, club and opponent.

    This is a local id (1, 2, 3, ...). It is NOT the StatsBomb match_id,
    which is added separately by add_statsbomb_match_id.
    """
    dataframe = dataframe.sort_values(["date", "goal_minute"], kind="stable").reset_index(drop=True)
    match_keys = ["date", "club", "opponent"]
    dataframe["goal_match_id"] = dataframe.groupby(match_keys, sort=False).ngroup() + 1
    dataframe["goal_number_in_match"] = dataframe.groupby("goal_match_id").cumcount() + 1
    return dataframe


def add_statsbomb_match_id(dataframe, statsbomb_matches_path):
    """
    Add 'match_id' = the StatsBomb match id, so this file joins directly with
    messi_league_matches.csv on match_id.

    Only league goals (La Liga, Ligue 1) can have one. Cup goals, and league
    matches missing from StatsBomb open data, keep an empty value.
    The join key is (date, competition) because the two sources spell
    opponents differently.
    """
    statsbomb_matches = pd.read_csv(statsbomb_matches_path, parse_dates=["date"])
    statsbomb_matches["competition"] = statsbomb_matches["competition"].map(STATSBOMB_COMPETITION_MAP)
    statsbomb_keys = statsbomb_matches[["date", "competition", "match_id"]]

    # many_to_one: several goals can share a match, but each match appears once in StatsBomb.
    dataframe = dataframe.merge(
        statsbomb_keys, on=["date", "competition"], how="left", validate="many_to_one"
    )
    dataframe["match_id"] = dataframe["match_id"].astype("Int64")  # integer that allows NaN

    is_league_goal = dataframe["competition"].isin(STATSBOMB_COMPETITION_MAP.values())
    unmatched = dataframe[is_league_goal & dataframe["match_id"].isna()]
    print(
        f"League goals: {int(is_league_goal.sum())}, "
        f"with a StatsBomb match_id: {int(dataframe['match_id'].notna().sum())}"
    )
    if len(unmatched) > 0:
        print("League goals with no StatsBomb match (kept, match_id left empty):")
        print(unmatched[["date", "competition", "opponent", "goal_minute"]].to_string(index=False))
    return dataframe
 
 
def reorder_columns(dataframe):
    """Put the columns in a readable order."""
    ordered_columns = [
        "goal_match_id", "match_id", "goal_number_in_match",
        "date", "year", "month", "season", "goal_decade",
        "club", "competition", "stage_name", "matchday", "opponent", "venue", "is_home_goal",
        "player_position", "goal_minute", "goal_minute_bucket",
        "goal_type", "goal_type_raw", "assist_player", "has_assist",
        "team_goals_before", "opponent_goals_before", "score_diff_before", "game_state_before",
        "team_goals_after", "opponent_goals_after",
        "team_goals_final", "opponent_goals_final", "match_result",
    ]
    return dataframe[ordered_columns]
 
 
# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
 
def run_sanity_checks(dataframe):
    """
    Print checks that should all pass. Raises AssertionError on a real problem
    and prints warnings for things worth a human look.
    """
    # 1. Row count must not change during cleaning.
    print(f"Rows: {len(dataframe)}")
 
    # 2. The goal cannot be scored after the final whistle score is exceeded.
    too_many_goals = dataframe["team_goals_after"] > dataframe["team_goals_final"]
    print(f"Rows where team goals after > final team goals: {int(too_many_goals.sum())}")
 
    too_many_opponent_goals = dataframe["opponent_goals_after"] > dataframe["opponent_goals_final"]
    print(f"Rows where opponent goals after > final opponent goals: {int(too_many_opponent_goals.sum())}")
 
    # 3. Before-score can never be negative.
    negative_before = dataframe["team_goals_before"] < 0
    print(f"Rows with negative 'team_goals_before': {int(negative_before.sum())}")
 
    # 4. is_home_goal should agree with venue.
    disagree = dataframe["is_home_goal"] != (dataframe["venue"] == "Home")
    print(f"Rows where is_home_goal disagrees with venue: {int(disagree.sum())}")
 
    # 5. Messi cannot score more goals in a match than his team's final total.
    goals_per_match = dataframe.groupby("goal_match_id").size()
    team_final_per_match = dataframe.groupby("goal_match_id")["team_goals_final"].first()
    impossible = (goals_per_match > team_final_per_match).sum()
    print(f"Matches where Messi goals > team final goals: {int(impossible)}")
 
    # 6. Matchday should only be missing for cup rounds.
    league_without_matchday = (
        (dataframe["stage_name"] == "League") & dataframe["matchday"].isna()
    ).sum()
    print(f"League rows without matchday: {int(league_without_matchday)}")

    # 7. A StatsBomb match_id must belong to exactly one local goal_match_id.
    linked = dataframe.dropna(subset=["match_id"])
    ids_per_statsbomb_match = linked.groupby("match_id")["goal_match_id"].nunique()
    print(f"StatsBomb match_id linked to more than one match: {int((ids_per_statsbomb_match > 1).sum())}")
 
 
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
 
def clean_messi_goals(input_path, statsbomb_matches_path, output_path):
    """Run every cleaning step in order and save the result."""
    dataframe = load_raw_data(input_path)
    dataframe = remove_excluded_clubs(dataframe)
    dataframe = clean_dates(dataframe)
    dataframe = clean_stage(dataframe)
    dataframe = clean_scores(dataframe)
    dataframe = clean_assists(dataframe)
    dataframe = clean_goal_types(dataframe)
    dataframe = clean_flags_and_categories(dataframe)
    dataframe = add_goal_match_id(dataframe)
    dataframe = add_statsbomb_match_id(dataframe, statsbomb_matches_path)
    dataframe = reorder_columns(dataframe)
 
    run_sanity_checks(dataframe)
 
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    print(f"Saved clean data to {output_path}")
    return dataframe
 
 
def parse_arguments():
    parser = argparse.ArgumentParser(description="Clean the Messi goals dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--statsbomb-matches", type=Path, default=DEFAULT_STATSBOMB_MATCHES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()
 
 
if __name__ == "__main__":
    arguments = parse_arguments()
    clean_messi_goals(arguments.input, arguments.statsbomb_matches, arguments.output)
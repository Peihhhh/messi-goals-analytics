"""Helper functions shared by the EDA notebook and the hypothesis-testing script."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MATCHES_FILE = Path("data/clean/messi_league_matches.csv")
SHOTS_FILE = Path("data/clean/messi_league_shots.csv")

LATE_BARCA_FIRST_SEASON = 2018      # first season of the "late Barca" period (2018-19)
MIN_SEASON_MINUTES = 900            # seasons with fewer minutes are too noisy for per-90 trends
N_BOOTSTRAP = 2000                  # resamples for the 95% confidence intervals
RANDOM_SEED = 42

PERIOD_ORDER = ["Barca 2004-18", "Barca 2018-21", "PSG 2021-23"]
LEAGUE_ORDER = ["LaLiga", "Ligue 1"]

# Colors follow the entity (same period / league = same color in every chart).
PERIOD_COLORS = {"Barca 2004-18": "#2a78d6", "PSG 2021-23": "#eb6834", "Barca 2018-21": "#1baf7a"}
LEAGUE_COLORS = {"LaLiga": "#2a78d6", "Ligue 1": "#eb6834"}
SECOND_SERIES_COLOR = "#eb6834"
GOAL_COLOR = "#2a78d6"
MISS_COLOR = "#b5b3ad"
TEXT_PRIMARY, TEXT_SECONDARY, GRID_COLOR, AXIS_COLOR = "#0b0b0b", "#52514e", "#e5e4df", "#c3c2b7"

# StatsBomb pitch: 120 x 80, Messi's team always attacks towards x = 120.
PENALTY_BOX = {"x_min": 102, "y_min": 18, "y_max": 62}
PITCH_VIEW_START = 75   # show the pitch from x = 75 to the goal line (almost every shot is in this zone)

# (column in summarize(), label, numerator, denominator, scale) for the period comparison.
PERIOD_METRICS = [
    ("shots_p90", "Cú sút / 90'", "shots", "minutes_played", 90),
    ("npxg_p90", "npxG / 90'", "npxg", "minutes_played", 90),
    ("np_goals_p90", "Bàn không pen / 90'", "non_penalty_goals", "minutes_played", 90),
    ("npxg_per_shot", "npxG / cú sút", "npxg", "non_penalty_shots", 1),
]


def apply_plot_style():
    """Matplotlib style used by every chart in the project."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.edgecolor": AXIS_COLOR,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "legend.frameon": False,
        "lines.linewidth": 2,
    })


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def season_start_year(season):
    """'2018-19' -> 2018."""
    return int(str(season)[:4])


def label_period(competition, season):
    """Put a match or shot into one of the three comparison periods."""
    if competition == "Ligue 1":
        return "PSG 2021-23"
    if season_start_year(season) >= LATE_BARCA_FIRST_SEASON:
        return "Barca 2018-21"
    return "Barca 2004-18"


def add_period(dataframe):
    periods = [label_period(c, s) for c, s in zip(dataframe["competition"], dataframe["season"])]
    dataframe["period"] = pd.Categorical(periods, categories=PERIOD_ORDER, ordered=True)
    return dataframe


def load_data(project_root):
    """Read the cleaned StatsBomb match and shot tables and attach the comparison period."""
    matches = pd.read_csv(project_root / MATCHES_FILE, parse_dates=["date"])
    shots = pd.read_csv(project_root / SHOTS_FILE, parse_dates=["date"])
    # In the shot table 'period' means the half (1, 2); rename it so it does not clash with our 'period'.
    shots = shots.rename(columns={"period": "half"})

    matches = add_period(matches).sort_values("date").reset_index(drop=True)
    shots = add_period(shots)
    return matches, shots


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def add_rate_columns(totals):
    """Per-90 and per-shot rates from summed totals (sum first, divide after)."""
    per_90 = 90 / totals["minutes"]
    totals["goals_p90"] = totals["goals"] * per_90
    totals["np_goals_p90"] = totals["np_goals"] * per_90
    totals["shots_p90"] = totals["shots"] * per_90
    totals["npxg_p90"] = totals["npxg"] * per_90
    totals["npxg_per_shot"] = totals["npxg"] / totals["np_shots"]
    totals["np_conversion"] = totals["np_goals"] / totals["np_shots"]
    totals["np_goals_minus_npxg"] = totals["np_goals"] - totals["npxg"]
    return totals


def summarize(matches, by):
    """Totals and rates for each group in `by`."""
    totals = matches.groupby(by, observed=True).agg(
        matches=("match_id", "size"),
        starts=("started", "sum"),
        minutes=("minutes_played", "sum"),
        goals=("goals", "sum"),
        np_goals=("non_penalty_goals", "sum"),
        shots=("shots", "sum"),
        np_shots=("non_penalty_shots", "sum"),
        npxg=("npxg", "sum"),
    )
    return add_rate_columns(totals)


def bootstrap_ratio_ci(matches, numerator, denominator, scale=1.0):
    """
    95% CI for sum(numerator) / sum(denominator) * scale, resampling whole matches.
    Resampling matches (not shots) keeps the shots of one match together.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    top = matches[numerator].to_numpy(dtype=float)
    bottom = matches[denominator].to_numpy(dtype=float)
    picks = rng.integers(0, len(top), size=(N_BOOTSTRAP, len(top)))
    ratios = top[picks].sum(axis=1) / bottom[picks].sum(axis=1) * scale
    return np.percentile(ratios, [2.5, 97.5])


def compare_periods(matches):
    """Long table: one row per (period, metric) with the value from summarize() and its 95% CI."""
    period_summary = summarize(matches, "period")
    rows = []
    for period, period_matches in matches.groupby("period", observed=True):
        for column, label, numerator, denominator, scale in PERIOD_METRICS:
            low, high = bootstrap_ratio_ci(period_matches, numerator, denominator, scale)
            rows.append({"period": period, "metric": label, "value": period_summary.loc[period, column],
                         "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def share_table(dataframe, row, column):
    """Percentage of each `column` value inside each `row` group (rows sum to 100)."""
    return pd.crosstab(dataframe[row], dataframe[column], normalize="index").mul(100).round(1)


def count_shares(values, groups, cap, group_order):
    """% of matches with 0, 1, ..., cap+ for each group."""
    capped = values.clip(upper=cap)
    shares = pd.crosstab(groups, capped, normalize="index").mul(100)
    shares.columns = [str(c) if c < cap else f"{cap}+" for c in shares.columns]
    return shares.reindex(group_order)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_grouped_shares(ax, shares, colors, title, xlabel):
    """Grouped bars: one group per column of `shares`, one bar per row (series)."""
    n_series = len(shares.index)
    width = 0.8 / n_series
    positions = np.arange(len(shares.columns))
    for i, series in enumerate(shares.index):
        ax.bar(positions + (i - (n_series - 1) / 2) * width, shares.loc[series], width=width * 0.92,
               color=colors[series], label=series)
    ax.set_xticks(positions, shares.columns)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("% trong nhóm")
    ax.grid(axis="x", visible=False)
    ax.legend()


def draw_attacking_zone(ax):
    """Attacking end of a StatsBomb pitch, drawn vertically (goal at the top)."""
    line = {"color": AXIS_COLOR, "linewidth": 1}
    ax.plot([0, 0, 80, 80], [PITCH_VIEW_START, 120, 120, PITCH_VIEW_START], **line)   # side + goal lines
    ax.add_patch(Rectangle((18, 102), 44, 18, fill=False, **line))                   # penalty box
    ax.add_patch(Rectangle((30, 114), 20, 6, fill=False, **line))                    # six-yard box
    ax.add_patch(Rectangle((36, 120), 8, 1.5, fill=False, **line))                   # goal
    ax.add_patch(Arc((40, 108), 20, 20, theta1=217, theta2=323, **line))             # D
    ax.scatter([40], [108], s=4, color=AXIS_COLOR)                                   # penalty spot
    ax.set_xlim(-2, 82)
    ax.set_ylim(PITCH_VIEW_START, 123)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_shot_map(ax, period_shots, title):
    """Misses in gray, goals in blue; marker area grows with xG."""
    is_goal = period_shots["is_goal"]
    for subset, color, z in [(period_shots[~is_goal], MISS_COLOR, 2), (period_shots[is_goal], GOAL_COLOR, 3)]:
        ax.scatter(subset["y"], subset["x"], s=8 + 120 * subset["xg"], color=color, alpha=0.6,
                   edgecolor="white", linewidth=0.4, zorder=z)
    draw_attacking_zone(ax)
    ax.set_title(title, fontsize=10)
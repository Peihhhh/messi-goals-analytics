"""Statistical tests for hypotheses H1-H4 (Messi at Barca vs PSG)."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.eda_utils import RANDOM_SEED, N_BOOTSTRAP


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASELINE_PERIOD = "Barca 2018-21"    # closest in age to PSG, so the fairest comparison
PSG_PERIOD = "PSG 2021-23"
ALPHA = 0.05
N_PERMUTATIONS = 5000
N_SIMULATIONS = 10000
# Poisson assumes variance = mean. Above this Pearson dispersion we switch to Negative Binomial.
OVERDISPERSION_THRESHOLD = 1.5
AGE_CENTER = 30                      # centering age keeps age and age^2 from being almost collinear


# ---------------------------------------------------------------------------
# Data selection
# ---------------------------------------------------------------------------

def main_comparison(dataframe):
    """Keep only Barca 2018-21 and PSG 2021-23 and add is_psg (0/1)."""
    selected = dataframe[dataframe["period"].isin([BASELINE_PERIOD, PSG_PERIOD])].copy()
    selected["is_psg"] = (selected["period"] == PSG_PERIOD).astype(int)
    return selected


def non_penalty_shots(shots):
    return shots[~shots["is_penalty"]].copy()


# ---------------------------------------------------------------------------
# Rate comparison (H1): permutation test + bootstrap CI
# ---------------------------------------------------------------------------

def rate_of_group(numerators, denominators, mask, scale):
    return numerators[mask].sum() / denominators[mask].sum() * scale


def rate_ratio_psg_vs_baseline(numerators, denominators, is_psg, scale):
    return rate_of_group(numerators, denominators, is_psg, scale) / rate_of_group(
        numerators, denominators, ~is_psg, scale
    )


def permutation_test_rate_ratio(matches, numerator, denominator, scale=90):
    """
    Two-sided permutation test for rate(PSG) / rate(Barca 2018-21).

    Under H0 the period label does not matter, so shuffling which matches are
    "PSG" should give ratios as extreme as the real one fairly often.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    top = matches[numerator].to_numpy(dtype=float)
    bottom = matches[denominator].to_numpy(dtype=float)
    is_psg = matches["is_psg"].to_numpy(dtype=bool)

    observed = rate_ratio_psg_vs_baseline(top, bottom, is_psg, scale)
    shuffled = np.array([
        rate_ratio_psg_vs_baseline(top, bottom, rng.permutation(is_psg), scale)
        for _ in range(N_PERMUTATIONS)
    ])
    # Compare on the log scale so a ratio of 0.5 and of 2 count as equally extreme.
    p_value = (np.sum(np.abs(np.log(shuffled)) >= abs(np.log(observed))) + 1) / (N_PERMUTATIONS + 1)
    return observed, p_value


def bootstrap_rate_ratio_ci(matches, numerator, denominator, scale=90):
    """95% CI for rate(PSG) / rate(Barca 2018-21), resampling matches inside each period."""
    rng = np.random.default_rng(RANDOM_SEED)

    def resampled_rates(period_matches):
        top = period_matches[numerator].to_numpy(dtype=float)
        bottom = period_matches[denominator].to_numpy(dtype=float)
        picks = rng.integers(0, len(top), size=(N_BOOTSTRAP, len(top)))
        return top[picks].sum(axis=1) / bottom[picks].sum(axis=1) * scale

    psg_rates = resampled_rates(matches[matches["is_psg"] == 1])
    baseline_rates = resampled_rates(matches[matches["is_psg"] == 0])
    return np.percentile(psg_rates / baseline_rates, [2.5, 97.5])


def compare_shot_xg(np_shots):
    """Mann-Whitney U on the xG of single non-penalty shots (is a PSG shot a better chance?)."""
    psg_xg = np_shots.loc[np_shots["is_psg"] == 1, "xg"]
    baseline_xg = np_shots.loc[np_shots["is_psg"] == 0, "xg"]
    result = stats.mannwhitneyu(psg_xg, baseline_xg, alternative="two-sided")
    return {"median_psg": psg_xg.median(), "median_baseline": baseline_xg.median(), "p_value": result.pvalue}


# ---------------------------------------------------------------------------
# Count models (H2, H4)
# ---------------------------------------------------------------------------

def fit_count_model(data, formula, exposure_column="minutes_played"):
    """
    Poisson regression with log(minutes) as offset, so coefficients are effects on the
    rate per minute. Falls back to Negative Binomial when the counts are overdispersed.
    """
    offset = np.log(data[exposure_column])
    poisson = smf.glm(formula, data=data, family=sm.families.Poisson(), offset=offset).fit()
    dispersion = poisson.pearson_chi2 / poisson.df_resid
    if dispersion <= OVERDISPERSION_THRESHOLD:
        return poisson, "Poisson", dispersion
    negative_binomial = smf.negativebinomial(formula, data=data, offset=offset).fit(disp=0)
    return negative_binomial, "Negative Binomial", dispersion


def exponentiated_effect(model, term):
    """exp(coefficient) with its 95% CI and p-value: a rate ratio or an odds ratio."""
    low, high = model.conf_int().loc[term]
    return {
        "estimate": float(np.exp(model.params[term])),
        "ci_low": float(np.exp(low)),
        "ci_high": float(np.exp(high)),
        "p_value": float(model.pvalues[term]),
    }


def add_age_terms(matches):
    matches = matches.copy()
    matches["age_c"] = matches["age"] - AGE_CENTER
    matches["is_psg"] = (matches["competition"] == "Ligue 1").astype(int)
    return matches


# ---------------------------------------------------------------------------
# Finishing (H3)
# ---------------------------------------------------------------------------

def fit_finishing_model(np_shots):
    """
    Logistic regression goal ~ logit(xG) + is_psg on non-penalty shots.

    logit(xG) controls for chance quality, so is_psg measures finishing at the
    same xG. Standard errors are clustered by match because shots in one match
    are not independent.
    """
    data = np_shots.copy()
    xg = data["xg"].clip(1e-4, 1 - 1e-4)
    data["logit_xg"] = np.log(xg / (1 - xg))
    data["goal"] = data["is_goal"].astype(int)
    return smf.glm("goal ~ logit_xg + is_psg", data=data, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": data["match_id"]}
    )


def simulate_goals_from_xg(xg_values, seed=RANDOM_SEED):
    """Goals an average finisher would score from these shots: each shot is a coin flip with p = xG."""
    rng = np.random.default_rng(seed)
    xg_values = np.asarray(xg_values, dtype=float)
    return (rng.random((N_SIMULATIONS, len(xg_values))) < xg_values).sum(axis=1)


def finishing_vs_xg(np_shots):
    """For each period: actual goals, expected goals, and how unusual the gap is for an average finisher."""
    rows = []
    for period, period_shots in np_shots.groupby("period", observed=True):
        simulated = simulate_goals_from_xg(period_shots["xg"])
        actual = int(period_shots["is_goal"].sum())
        rows.append({
            "period": period,
            "shots": len(period_shots),
            "goals": actual,
            "npxg": period_shots["xg"].sum(),
            "goals_minus_npxg": actual - period_shots["xg"].sum(),
            "share_sims_at_least_actual": (simulated >= actual).mean(),
            "share_sims_at_most_actual": (simulated <= actual).mean(),
        })
    return pd.DataFrame(rows).set_index("period")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def result_row(hypothesis, claim, method, effect_label, effect):
    return {"hypothesis": hypothesis, "claim": claim, "method": method, "effect": effect_label, **effect}


def add_holm_and_conclusion(results):
    """Holm-adjust the p-values of the main tests and write a one-line verdict."""
    results = results.copy()
    results["p_holm"] = multipletests(results["p_value"], method="holm")[1]
    results["significant"] = results["p_holm"] < ALPHA
    results["direction"] = np.where(results["estimate"] < 1, "PSG thấp hơn", "PSG cao hơn")
    results["conclusion"] = np.where(
        results["significant"], "Bác bỏ H0 (" + results["direction"] + ")", "Chưa đủ bằng chứng bác bỏ H0"
    )
    return results


def format_effect(effect, digits=2):
    return (f"{effect['estimate']:.{digits}f} (95% CI {effect['ci_low']:.{digits}f}–{effect['ci_high']:.{digits}f}), "
            f"p = {effect['p_value']:.4f}")

"""
Klimek et al. (2012) Election Fraud Detection
Faithful port of the EFToolkit R implementation (Mebane & Egami, 2014)
which is itself a pure conversion of the original Matlab ElectionFritter.m.

Reference: Klimek, P., Yegorov, Y., Hanel, R., & Thurner, S. (2012).
Statistical detection of systematic election irregularities.
PNAS, 109(41), 16469-16473.

INPUT FILE: klimek_template.csv (must be in the same directory as this script)

    Required CSV columns:
        precinct_id   : any string/int identifier for the precinct
        total_votes   : total ballots cast in that precinct
        registered    : total registered voters in that precinct (used for turnout)
        winning_votes : votes received by the winning party in that precinct

    Example row: "Cook_001", 412, 604, 287

KEY IMPLEMENTATION NOTES (differences from naive implementations):
    - fi and fe are NOT independent probabilities. The model draws:
          f2 = fi + fe  (total fraud probability per precinct)
          f1 = fi / (fi + fe)  (fraction of fraud that is incremental)
      A precinct gets fraud with prob f2; given fraud, it is incremental
      with prob f1 and extreme with prob (1-f1). They are mutually exclusive.
    - theta = (sigma_R_v)^(1/4), not sqrt(sigma_R_v)
    - xi drawn from |N(0, theta)|, yi from 1 - |N(1, 0.075)|
    - Vote share drawn as 2-candidate normalized vector, not single normal
    - S statistic denominator is (Obs + 1), not Obs, to avoid div-by-zero
    - v_bar estimated by first downslope in histogram, not argmax
    - Grid autoscales: initial scan expands if best fit is near boundary

OUTPUTS:
    - Console table of best-fit (fi, fe, alpha) parameters and S value
    - 2D fingerprint heatmap (turnout x vote_share)
    - S surface plot over (fi, fe) at best alpha
    - CSV of all S values across the parameter grid
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from itertools import product
import argparse
import os
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 1. DATA LOADING & VALIDATION
# ---------------------------------------------------------------------------

REQUIRED_COLS = {"precinct_id", "total_votes", "winning_votes", "registered"}


def load_data(path: str) -> pd.DataFrame:
    # Required columns: precinct_id, total_votes, registered, winning_votes
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Input file '{path}' not found. "
            "Place klimek_template.csv in the same directory as this script.\n"
            "Required columns: precinct_id, total_votes, registered, winning_votes"
        )
    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df


def validate_and_derive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_votes"]   = df["total_votes"].astype(int)
    df["winning_votes"] = df["winning_votes"].astype(int)
    df["registered"]    = df["registered"].astype(int)

    bad = (
        (df["winning_votes"] > df["total_votes"]) |
        (df["total_votes"]   <= 0)                |
        (df["registered"]    <= 0)                |
        (df["winning_votes"] <= 0)
    )
    n_bad = bad.sum()
    if n_bad:
        print(f"[WARN] Dropping {n_bad} rows with impossible vote counts.")
        df = df[~bad].copy()

    # Per Klimek et al. (2012) and ETA methodology:
    # Drop precincts with fewer than 100 registered voters to avoid
    # extreme turnout/vote rates as artifacts from very small communities.
    n_before = len(df)
    df = df[df["registered"] >= 100].copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"[INFO] Dropped {n_dropped} precincts with <100 registered voters.")

    df["vote_share"] = df["winning_votes"] / df["total_votes"]

    # Per Klimek et al. (2012) and ETA methodology:
    # Precincts reporting >100% turnout are capped at 100% rather than dropped.
    raw_turnout = df["total_votes"] / df["registered"]
    n_over = (raw_turnout > 1.0).sum()
    if n_over:
        print(f"[INFO] Capping turnout at 100% for {n_over} precincts "
              f"that reported >100% turnout.")
    df["turnout"] = raw_turnout.clip(0, 1)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. HISTOGRAM BINS (adaptive per R implementation)
# ---------------------------------------------------------------------------

def make_bins(n: int, thres: float = 5.0) -> np.ndarray:
    """
    Adaptive bin edges matching R implementation.
    x0 = -0.005*(5/thres), h = 0.01*(5/thres)
    thres is reduced if bin count would exceed N*0.8.
    """
    while True:
        x0 = -0.005 * (5.0 / thres)
        h  =  0.010 * (5.0 / thres)
        if 0.8 * n > 1.0 / h:
            break
        if thres > 0.5:
            thres -= 0.5
        else:
            break
    bins = np.arange(x0, 1.0 - x0 + h * 0.5, h)
    return bins


def empirical_counts(vote_shares: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Raw histogram counts (not normalized) — matches R hist()$counts."""
    counts, _ = np.histogram(vote_shares, bins=bins)
    return counts.astype(float)


# ---------------------------------------------------------------------------
# 3. DISTRIBUTION PARAMETER ESTIMATION (faithful to R Estimate() function)
# ---------------------------------------------------------------------------

def estimate_params(df: pd.DataFrame, bins: np.ndarray, thres: float = 5.0) -> dict:
    """
    Port of R Estimate() function.

    v_bar (lambda_fraud[1]): vote share at first downslope > thres after bin 10
    p_Att: turnout at first downslope > thres after bin 15
    stdAtt: RMS turnout deviation from p_Att for double-clean precincts
            (vi < v_bar AND ai < p_Att)
    theta: (sigma_R_v)^(1/4) — NOTE this is fourth-root, not sqrt
    sigma_fraud: sqrt(2 * mean((vi - v_bar)^2 for vi < v_bar))
                 applied symmetrically to both candidates
    """
    vote_shares = df["vote_share"].values
    turnouts    = df["turnout"].values

    # --- Estimate v_bar (lambda_fraud) ---
    n_v = empirical_counts(vote_shares, bins)
    sl_v = np.diff(n_v[9:])  # diff from bin 10 onwards (0-indexed: [9:])

    # Compute argmax and mean as sanity-check anchors
    v_bar_argmax = float(bins[np.argmax(n_v)])
    v_bar_mean   = float(np.mean(vote_shares))

    lfthres = thres
    v_bar = None
    while lfthres > 1.0:
        idx = np.where(sl_v < -lfthres)[0]
        if len(idx) > 0:
            bin_idx = idx[0] + 9  # +9 because we started diff from bin 10
            candidate = float(bins[bin_idx])
            # Sanity check: slope-change result must be within 10pp of the
            # argmax AND at least 90% of the mean vote share.
            # Tighter than the original 20pp to catch left-shoulder artifacts
            # in narrow right-skewed distributions (e.g. ballot measures).
            if (abs(candidate - v_bar_argmax) <= 0.10 and
                    candidate >= v_bar_mean * 0.90):
                v_bar = candidate
            break
        lfthres -= 0.5

    if v_bar is None:
        # Fallback: use argmax of histogram
        v_bar = v_bar_argmax
        print(f"[INFO] v_bar slope-change fallback to argmax: {v_bar:.3f} "
              f"(mean={v_bar_mean:.3f})")

    # Secondary sanity: if v_bar is still implausibly far below the mean,
    # fall back to argmax or mean as anchor.
    if v_bar < v_bar_mean * 0.80:
        old_v_bar = v_bar
        v_bar = v_bar_argmax if v_bar_argmax >= v_bar_mean * 0.80 else v_bar_mean
        print(f"[INFO] v_bar overridden from {old_v_bar:.3f} to {v_bar:.3f} "
              f"(mean={v_bar_mean:.3f}, argmax={v_bar_argmax:.3f})")

    lambda_fraud = np.array([v_bar, 1.0 - v_bar])

    # --- Estimate p_Att (modal turnout) ---
    a_bar_argmax = float(bins[np.argmax(empirical_counts(turnouts, bins))])
    a_bar_mean   = float(np.mean(turnouts))

    n_a = empirical_counts(turnouts, bins)

    sl_a = np.diff(n_a[14:])  # diff from bin 15 onwards

    p_Att = None
    lfthres2 = thres
    while lfthres2 > 1.0:
        idx = np.where(sl_a < -lfthres2)[0]
        if len(idx) > 0:
            bin_idx = idx[0] + 14
            candidate = float(bins[bin_idx])
            if abs(candidate - a_bar_argmax) <= 0.20 and candidate >= 0.10:
                p_Att = candidate
            break
        lfthres2 -= 0.5

    if p_Att is None:
        p_Att = a_bar_argmax
        print(f"[INFO] p_Att slope-change fallback to argmax: {p_Att:.3f} "
              f"(mean={a_bar_mean:.3f})")

    if p_Att < a_bar_mean * 0.5:
        p_Att = a_bar_argmax if a_bar_argmax >= a_bar_mean * 0.5 else a_bar_mean
        print(f"[INFO] p_Att overridden to {p_Att:.3f} (was implausibly low; "
              f"mean={a_bar_mean:.3f}, argmax={a_bar_argmax:.3f})")

    # --- stdAtt: from double-clean precincts (vi < v_bar AND ai < p_Att) ---
    s1 = turnouts[(vote_shares < v_bar) & (turnouts < p_Att)] - p_Att
    stdAtt = float(np.sqrt(np.sum(s1**2) / max(len(s1), 1)))

    # --- sigma_fraud: left-side RMS (applied symmetrically) ---
    s2 = vote_shares[vote_shares < v_bar] - v_bar
    sigma_fraud = float(np.sqrt(2.0 * np.sum(s2**2) / max(len(s2), 1)))

    # --- theta: fourth-root of right-side variance ---
    # R: theta <- sqrt(sqrt(sum(s3^2)/length(s3)))
    # where s3 = vi - v_max for vi > v_max (v_max = argmax of histogram)
    v_max = bins[np.argmax(n_v)]
    s3 = vote_shares[vote_shares > v_max] - v_max
    rms_right = np.sqrt(np.sum(s3**2) / max(len(s3), 1))
    theta = float(np.sqrt(rms_right))  # fourth root = sqrt(sqrt(...))

    return {
        "lambda_fraud": lambda_fraud,   # [v_bar, 1-v_bar]
        "v_bar":        v_bar,
        "p_Att":        p_Att,
        "stdAtt":       stdAtt,
        "sigma_fraud":  sigma_fraud,    # applied symmetrically to both candidates
        "theta":        theta,          # (sigma_R_v)^(1/4)
        "sigma_x":      0.075,          # fixed per paper
    }


# ---------------------------------------------------------------------------
# 4. MODEL SIMULATION (faithful port of R Sim_Vote() + Sim.Histo())
# ---------------------------------------------------------------------------

def simulate_votes(params: dict,
                   n_voters: np.ndarray,
                   n_sim: int,
                   f2: float,
                   f1: float,
                   alpha: float) -> np.ndarray:
    """
    Port of R Sim_Vote() function.

    f2 = fi + fe  (total fraud probability)
    f1 = fi / (fi + fe)  (fraction that is incremental; remainder is extreme)

    Returns array of winner vote shares for n_sim simulated precincts.

    Key differences from naive implementation:
    - Vote shares drawn as 2-candidate normalized vector (not single normal)
    - Incremental and extreme fraud are MUTUALLY EXCLUSIVE per precinct
    - xi ~ |N(0, theta)| where theta = (sigma_R_v)^(1/4)
    - yi ~ 1 - |N(1, 0.075)|  (NOT uniform on [0.95,1])
    - Fraud adds to winner from non-attendees AND recasts opponent votes
    """
    lambda_fraud = params["lambda_fraud"]
    p_Att        = params["p_Att"]
    stdAtt       = params["stdAtt"]
    sigma_fraud  = params["sigma_fraud"]
    theta        = params["theta"]
    sigma_x      = params["sigma_x"]

    # i) Sample electorates
    idx      = np.random.randint(0, len(n_voters), size=n_sim)
    NVoters  = n_voters[idx].astype(float)

    # ii) Turnout ~ N(p_Att, stdAtt), rejection-sampled to (0,1)
    a = np.random.normal(p_Att, stdAtt, n_sim)
    bad = (a < 0) | (a > 1)
    while bad.any():
        a[bad] = np.random.normal(p_Att, stdAtt, bad.sum())
        bad = (a < 0) | (a > 1)

    # iii) 2-candidate vote share vector, normalized to sum to 1
    # l[:,0] = winner share, l[:,1] = loser share
    l = np.column_stack([
        np.random.normal(lambda_fraud[0], sigma_fraud, n_sim),
        np.random.normal(lambda_fraud[1], sigma_fraud, n_sim),
    ])
    row_sums = l.sum(axis=1, keepdims=True)
    l = l / row_sums
    # Rejection sample rows where either share is out of (0,1)
    bad = np.any((l < 0) | (l > 1), axis=1)
    while bad.any():
        nb = bad.sum()
        l_new = np.column_stack([
            np.random.normal(lambda_fraud[0], sigma_fraud, nb),
            np.random.normal(lambda_fraud[1], sigma_fraud, nb),
        ])
        l_new = l_new / l_new.sum(axis=1, keepdims=True)
        l[bad] = l_new
        bad = np.any((l < 0) | (l > 1), axis=1)

    # Baseline vote counts (rounded per R implementation)
    FraudVotes = np.round(NVoters[:, None] * a[:, None] * l)  # shape (n_sim, 2)

    # iv/v) Fraud assignment — mutually exclusive per precinct
    fraud_flag = np.random.random(n_sim) < f2       # precinct gets any fraud
    sub_flag   = np.random.random(n_sim) < f1       # of those: incremental
    inc_flag   = fraud_flag & sub_flag               # incremental fraud precincts
    ext_flag   = fraud_flag & ~sub_flag              # extreme fraud precincts

    # Fraud intensities
    # xi ~ |N(0, theta)| for incremental, accepted in (0,1)
    us = np.zeros(n_sim)
    n_inc = inc_flag.sum()
    if n_inc > 0:
        xi = np.abs(np.random.normal(0, theta, n_inc * 5))
        xi = xi[(xi > 0) & (xi < 1)]
        while len(xi) < n_inc:
            xi = np.concatenate([xi,
                np.abs(np.random.normal(0, theta, n_inc))])
            xi = xi[(xi > 0) & (xi < 1)]
        us[inc_flag] = xi[:n_inc]

    # yi ~ 1 - |N(0, sigma_x)| for extreme, accepted in (0,1)
    # R code: abs(rnorm(n, sd=0.075)) subtracted from 1 — mean is 0, NOT 1.
    # This gives values tightly clustered near 1.0 (e.g. 0.92-0.99).
    n_ext = ext_flag.sum()
    if n_ext > 0:
        yi = 1.0 - np.abs(np.random.normal(0.0, sigma_x, n_ext * 5))
        yi = yi[(yi > 0) & (yi < 1)]
        while len(yi) < n_ext:
            yi = np.concatenate([yi,
                1.0 - np.abs(np.random.normal(0.0, sigma_x, n_ext))])
            yi = yi[(yi > 0) & (yi < 1)]
        us[ext_flag] = yi[:n_ext]

    cv0 = us ** alpha  # wrong-counting exponent

    # Apply fraud to winner votes:
    # from non-attendees: us * (NVoters - total_valid_votes)
    # from opponents:     cv0 * opponent_votes
    total_valid = FraudVotes[:, 0] + FraudVotes[:, 1]
    non_attendees = NVoters - total_valid
    FraudVotes[:, 0] += (np.floor(us * non_attendees) +
                         np.round(cv0 * FraudVotes[:, 1]))
    FraudVotes[:, 1] -= np.round(cv0 * FraudVotes[:, 1])
    FraudVotes = np.clip(FraudVotes, 0, None)

    # Winner vote share
    row_totals = FraudVotes.sum(axis=1)
    row_totals = np.where(row_totals <= 0, 1, row_totals)
    winner_shares = FraudVotes[:, 0] / row_totals

    return winner_shares


def model_counts(params: dict,
                 n_voters: np.ndarray,
                 bins: np.ndarray,
                 n_sim: int,
                 f2: float,
                 f1: float,
                 alpha: float) -> np.ndarray:
    """Simulate votes and return histogram counts over bins."""
    shares = simulate_votes(params, n_voters, n_sim, f2, f1, alpha)
    counts, _ = np.histogram(shares, bins=bins)
    return counts.astype(float)


# ---------------------------------------------------------------------------
# 5. S STATISTIC (with +1 smoothing per R implementation)
# ---------------------------------------------------------------------------

def compute_S(obs: np.ndarray, sim: np.ndarray) -> float:
    """
    S = sum((sim - obs)^2 / (obs + 1))

    Note: R uses (Obs.H.Vote + 1) in denominator — the +1 smoothing prevents
    division by zero without excluding any bins, and differs from the raw
    Eq. S3 which uses obs in the denominator.
    """
    return float(np.sum((sim - obs) ** 2 / (obs + 1.0)))


# ---------------------------------------------------------------------------
# 6. GRID SEARCH WITH AUTOSCALING
# ---------------------------------------------------------------------------

def build_grids(f2_max: float = 1.0,
                f2_step: float = 0.05,
                alpha_vals: np.ndarray = None) -> tuple:
    """
    Build search grids over f1 (0→1), f2 (0→f2_max), alpha.
    f1 = fi/(fi+fe), f2 = fi+fe
    """
    f1_grid    = np.arange(0.0, 1.01, 0.1)
    f2_grid    = np.arange(0.0, f2_max + f2_step * 0.5, f2_step)
    if alpha_vals is None:
        alpha_grid = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    else:
        alpha_grid = alpha_vals
    return f1_grid, f2_grid, alpha_grid


def grid_search(df: pd.DataFrame,
                params: dict,
                bins: np.ndarray,
                obs_counts: np.ndarray,
                n_sim: int = 5_000,
                n_realizations: int = 10,
                f2_max: float = 1.0,
                quick: bool = False) -> pd.DataFrame:
    """
    Search over (f1, f2, alpha) grid, averaging S over n_realizations.

    Autoscaling: if the best-fit f2 lands at the boundary of the grid,
    automatically expands and re-runs in that direction.

    f1 = fi/(fi+fe), f2 = fi+fe
    fi = f1 * f2, fe = (1-f1) * f2
    """
    n_voters = df["registered"].values

    if quick:
        f1_grid    = np.arange(0.0, 1.01, 0.2)
        f2_grid    = np.arange(0.0, f2_max + 0.1, 0.1)
        alpha_grid = np.array([0.5, 1.0, 2.0])
        n_realizations = 5
    else:
        f1_grid, f2_grid, alpha_grid = build_grids(f2_max=f2_max)

    for expansion in range(3):  # allow up to 3 autoscale expansions
        total = len(f1_grid) * len(f2_grid) * len(alpha_grid)
        print(f"[INFO] Grid: {len(f1_grid)} f1 × {len(f2_grid)} f2 × "
              f"{len(alpha_grid)} alpha = {total:,} cells × "
              f"{n_realizations} realizations")
        print(f"[INFO] f2 range: 0 → {f2_grid.max():.2f}  "
              f"(f2 = fi+fe; f1 = fi/(fi+fe))\n")

        results = []
        iterator = tqdm(list(product(f1_grid, f2_grid, alpha_grid)),
                        desc="Grid search", unit="cell")

        for f1, f2, alpha in iterator:
            s_vals = [
                compute_S(obs_counts,
                          model_counts(params, n_voters, bins,
                                       n_sim, f2, f1, alpha))
                for _ in range(n_realizations)
            ]
            fi = f1 * f2
            fe = (1.0 - f1) * f2
            results.append({
                "fi": round(fi, 4), "fe": round(fe, 4),
                "f1": round(f1, 4), "f2": round(f2, 4),
                "alpha": alpha,
                "S": float(np.mean(s_vals))
            })

        results_df = pd.DataFrame(results).sort_values("S").reset_index(drop=True)
        best = results_df.iloc[0]

        # --- Autoscale check ---
        best_f2 = best["f2"]
        at_boundary = best_f2 >= f2_grid.max() - 0.01

        if at_boundary and f2_grid.max() < 1.0:
            new_max = min(f2_grid.max() + 0.3, 1.0)
            print(f"\n[AUTOSCALE] Best f2={best_f2:.2f} is at grid boundary. "
                  f"Expanding f2 grid to {new_max:.2f}...\n")
            f2_step = 0.05 if not quick else 0.1
            f1_grid, f2_grid, alpha_grid = build_grids(
                f2_max=new_max, f2_step=f2_step)
        else:
            break  # no expansion needed

    return results_df


def refine_grid_search(df: pd.DataFrame,
                       params: dict,
                       bins: np.ndarray,
                       obs_counts: np.ndarray,
                       coarse_best: pd.Series,
                       n_sim: int = 5_000,
                       n_realizations: int = 100) -> pd.DataFrame:
    """
    Second-pass fine grid search zoomed in around the coarse best-fit.

    Builds a tight grid of ±0.15 around the best fi and fe, with step 0.01.
    Alpha grid is also narrowed to ±1.0 around the best alpha.
    Runs with higher realizations than the coarse pass for stability.
    """
    n_voters = df["registered"].values

    best_fi    = coarse_best["fi"]
    best_fe    = coarse_best["fe"]
    best_f2    = coarse_best["f2"]
    best_alpha = coarse_best["alpha"]

    # Build fine fi/fe grid: ±0.15 around best, step 0.01, clipped to [0,1]
    fi_min = max(0.0,  best_fi - 0.15)
    fi_max = min(1.0,  best_fi + 0.15)
    fe_min = max(0.0,  best_fe - 0.10)
    fe_max = min(1.0,  best_fe + 0.10)

    # Convert back to f1/f2 space for the grid
    # f2 = fi + fe, f1 = fi / f2
    f2_min = max(0.01, fi_min + fe_min)
    f2_max = min(1.0,  fi_max + fe_max)
    f2_grid = np.arange(f2_min, f2_max + 0.01, 0.01)

    # f1 range: fi/(fi+fe) for the corner combinations
    f1_candidates = []
    for fi in [fi_min, best_fi, fi_max]:
        for fe in [fe_min, best_fe, fe_max]:
            f2 = fi + fe
            if f2 > 0:
                f1_candidates.append(fi / f2)
    f1_min = max(0.0, min(f1_candidates) - 0.05)
    f1_max = min(1.0, max(f1_candidates) + 0.05)
    f1_grid = np.arange(f1_min, f1_max + 0.01, 0.02)

    # Alpha: ±1.0 around best, step 0.5
    alpha_min = max(0.5,  best_alpha - 1.0)
    alpha_max = min(5.0,  best_alpha + 1.0)
    alpha_grid = np.arange(alpha_min, alpha_max + 0.1, 0.5)

    total = len(f1_grid) * len(f2_grid) * len(alpha_grid)
    print(f"[REFINE] Fine grid: {len(f1_grid)} f1 × {len(f2_grid)} f2 × "
          f"{len(alpha_grid)} alpha = {total:,} cells × {n_realizations} realizations")
    print(f"[REFINE] fi range: {fi_min:.2f}->{fi_max:.2f}  "
          f"fe range: {fe_min:.2f}->{fe_max:.2f}  "
          f"alpha range: {alpha_min:.1f}->{alpha_max:.1f}\n")

    results = []
    iterator = tqdm(list(product(f1_grid, f2_grid, alpha_grid)),
                    desc="Fine grid", unit="cell")

    for f1, f2, alpha in iterator:
        fi = f1 * f2
        fe = (1.0 - f1) * f2
        # Skip if fi/fe are outside our target window
        if fi < fi_min - 0.01 or fi > fi_max + 0.01:
            continue
        if fe < fe_min - 0.01 or fe > fe_max + 0.01:
            continue
        s_vals = [
            compute_S(obs_counts,
                      model_counts(params, n_voters, bins,
                                   n_sim, f2, f1, alpha))
            for _ in range(n_realizations)
        ]
        results.append({
            "fi": round(fi, 4), "fe": round(fe, 4),
            "f1": round(f1, 4), "f2": round(f2, 4),
            "alpha": alpha,
            "S": float(np.mean(s_vals))
        })

    if not results:
        print("[WARN] Fine grid produced no results — returning coarse results.")
        return pd.DataFrame([coarse_best])

    return pd.DataFrame(results).sort_values("S").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7. VISUALIZATION
# ---------------------------------------------------------------------------

def plot_fingerprint(df: pd.DataFrame, output_dir: str = ".", label: str = "") -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    h, xedges, yedges = np.histogram2d(
        df["turnout"], df["vote_share"],
        bins=50, range=[[0, 1], [0, 1]]
    )
    h_norm = h / h.sum()
    im = ax.imshow(
        h_norm.T, origin="lower", extent=[0, 1, 0, 1],
        aspect="auto", cmap="hot_r", norm=mcolors.PowerNorm(gamma=0.4)
    )
    plt.colorbar(im, ax=ax, label="Fraction of precincts")
    ax.set_xlabel("Voter turnout", fontsize=12)
    ax.set_ylabel("Winning party vote share", fontsize=12)
    title = f"{label} — Election Fingerprint (Klimek et al.)" if label else "Election Fingerprint (Klimek et al.)"
    ax.set_title(title, fontsize=13)
    fname = f"{label}_fingerprint.png" if label else "fingerprint.png"
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[OUT] Fingerprint → {path}")
    plt.close(fig)


def plot_S_surface(results: pd.DataFrame, best_alpha: float,
                   output_dir: str = ".", label: str = "") -> None:
    sub = results[np.isclose(results["alpha"], best_alpha, atol=0.01)].copy()
    if sub.empty:
        print("[WARN] No results at best_alpha for surface plot.")
        return

    try:
        pivot = sub.pivot_table(index="fe", columns="fi", values="S", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(
            pivot.values, origin="lower",
            extent=[pivot.columns.min(), pivot.columns.max(),
                    pivot.index.min(),   pivot.index.max()],
            aspect="auto", cmap="viridis_r",
            norm=mcolors.PowerNorm(gamma=0.5)
        )
        plt.colorbar(im, ax=ax, label="S statistic")
        ax.set_xlabel("Incremental fraud fᵢ", fontsize=12)
        ax.set_ylabel("Extreme fraud fₑ", fontsize=12)
        title = f"{label} — S(fᵢ, fₑ) surface at α={best_alpha:.1f}" if label else f"S(fᵢ, fₑ) surface at α={best_alpha:.1f}"
        ax.set_title(title, fontsize=13)
        min_row = sub.loc[sub["S"].idxmin()]
        ax.plot(min_row["fi"], min_row["fe"], "r*", markersize=14,
                label=f"Min @ fᵢ={min_row['fi']:.3f}, fₑ={min_row['fe']:.3f}")
        ax.legend(fontsize=10)
        fname = f"{label}_S_surface.png" if label else "S_surface.png"
        path = os.path.join(output_dir, fname)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[OUT] S surface → {path}")
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] S surface plot failed: {e}")


def plot_pdf_comparison(df: pd.DataFrame,
                        params: dict, bins: np.ndarray,
                        obs_counts: np.ndarray,
                        f1: float, f2: float, alpha: float,
                        n_sim: int = 20_000,
                        output_dir: str = ".",
                        label: str = "") -> None:
    n_voters = df["registered"].values
    sim_counts = model_counts(params, n_voters, bins, n_sim, f2, f1, alpha)

    bin_centers = (bins[:-1] + bins[1:]) / 2
    obs_frac = obs_counts / obs_counts.sum()
    sim_frac = sim_counts / max(sim_counts.sum(), 1)

    fi = f1 * f2
    fe = (1.0 - f1) * f2

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(bin_centers, obs_frac, width=(bins[1]-bins[0]), alpha=0.6,
           color="steelblue", label="Empirical")
    ax.plot(bin_centers, sim_frac, color="crimson", linewidth=2,
            label=f"Model (fᵢ={fi:.3f}, fₑ={fe:.3f}, α={alpha:.1f})")
    ax.set_xlabel("Winning party vote share", fontsize=12)
    ax.set_ylabel("Fraction of precincts", fontsize=12)
    title = f"{label} — Empirical vs. Best-fit Model Distribution" if label else "Empirical vs. Best-fit Model Distribution"
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    fname = f"{label}_pdf_comparison.png" if label else "pdf_comparison.png"
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[OUT] PDF comparison → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. INTERPRETATION
# ---------------------------------------------------------------------------

def interpret(best: pd.Series, params: dict) -> str:
    fi    = best["fi"]
    fe    = best["fe"]
    f2    = best["f2"]
    alpha = best["alpha"]
    S     = best["S"]

    lines = [
        "",
        "=" * 65,
        "  FRAUD PARAMETER INTERPRETATION",
        "=" * 65,
        f"  Best-fit:  fᵢ={fi:.4f}  fₑ={fe:.4f}  α={alpha:.2f}  "
        f"(f2={f2:.3f})",
        f"  Minimum S statistic: {S:.6f}",
        "",
        f"  Distribution params: v̄={params['v_bar']:.3f}  "
        f"ā={params['p_Att']:.3f}  "
        f"θ={params['theta']:.4f}  "
        f"σ_fraud={params['sigma_fraud']:.4f}",
        "",
    ]

    # Incremental
    if fi < 0.02:
        lines.append(f"  fᵢ={fi:.4f} → Consistent with NO incremental fraud.")
    elif fi < 0.10:
        lines.append(f"  fᵢ={fi:.4f} → Low-level incremental signal.")
    elif fi < 0.25:
        lines.append(f"  fᵢ={fi:.4f} → SUSPICIOUS: meaningful incremental padding.")
    else:
        lines.append(f"  fᵢ={fi:.4f} → HIGH: strong incremental fraud signal.")

    # Extreme
    if fe < 0.01:
        lines.append(f"  fₑ={fe:.4f} → Consistent with NO extreme fraud.")
    elif fe < 0.05:
        lines.append(f"  fₑ={fe:.4f} → Low-level extreme fraud signal.")
    elif fe < 0.15:
        lines.append(f"  fₑ={fe:.4f} → SUSPICIOUS: notable ballot-box stuffing.")
    else:
        lines.append(f"  fₑ={fe:.4f} → HIGH: strong extreme fraud signal.")

    # Alpha
    if alpha < 1.0:
        lines.append(f"  α={alpha:.2f}  → Wrong-counting process dominates.")
    else:
        lines.append(f"  α={alpha:.2f}  → Urn stuffing mechanism dominates.")

    lines += [
        "",
        "  NOTE: Statistical signal only — not proof of fraud.",
        "  Paper finds fi~0 for all clean Western democracies.",
        "  Russia 2012: fi≈0.39, fe≈0.02. Uganda: fi≈0.49-0.83.",
        "=" * 65,
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 9. MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Klimek et al. (2012) election fraud detection."
    )
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Coarse grid for fast exploration.")
    parser.add_argument("--n-sim", type=int, default=5_000,
                        help="Monte Carlo draws per realization (default: 5000).")
    parser.add_argument("--realizations", type=int, default=10,
                        help="Realizations to average S over (default: 10).")
    args = parser.parse_args()

    # --- Output subfolder ---
    BASE_OUTPUT_DIR = r"D:\Klimek"
    print(f"\nOutputs will be saved to: {BASE_OUTPUT_DIR}\\<SubfolderName>")
    while True:
        subfolder = input("Enter subfolder name for this run: ").strip()
        if subfolder:
            break
        print("[WARN] Cannot be empty.")
    output_dir = os.path.join(BASE_OUTPUT_DIR, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output dir: {output_dir}\n")

    # --- Load data ---
    # Required CSV columns: precinct_id, total_votes, registered, winning_votes
    input_path = "klimek_template.csv"
    df_raw = load_data(input_path)
    df     = validate_and_derive(df_raw)
    print(f"[INFO] {len(df)} valid precincts loaded.\n")
    print(df[["precinct_id", "total_votes", "winning_votes",
              "vote_share", "turnout"]].head(10).to_string(index=False))
    print("...\n")

    # --- Bins and empirical distribution ---
    bins       = make_bins(len(df))
    obs_counts = empirical_counts(df["vote_share"].values, bins)

    # --- Estimate distribution parameters ---
    params = estimate_params(df, bins)
    print(f"[INFO] v̄={params['v_bar']:.3f}  ā={params['p_Att']:.3f}  "
          f"θ={params['theta']:.4f}  σ_fraud={params['sigma_fraud']:.4f}  "
          f"stdAtt={params['stdAtt']:.4f}\n")

    # --- Fingerprint ---
    plot_fingerprint(df, output_dir, label=subfolder)


    # --- Grid search ---
    results = grid_search(
        df, params, bins, obs_counts,
        n_sim=args.n_sim,
        n_realizations=args.realizations,
        quick=args.quick
    )

    # --- Save coarse results ---
    csv_path = os.path.join(output_dir, "S_grid_results_coarse.csv")
    results.to_csv(csv_path, index=False)
    print(f"\n[OUT] Coarse grid results → {csv_path}")

    coarse_best = results.iloc[0]
    print(f"\n  Coarse top 5 (lowest S):")
    print(results.head(5)[["fi","fe","f2","alpha","S"]].to_string(index=False))

    # --- Refinement pass ---
    if not args.quick:
        print("\n[INFO] Running fine grid refinement around coarse best-fit...\n")
        refined = refine_grid_search(
            df, params, bins, obs_counts,
            coarse_best=coarse_best,
            n_sim=args.n_sim,
            n_realizations=args.realizations
        )
        refined_csv = os.path.join(output_dir, "S_grid_results_refined.csv")
        refined.to_csv(refined_csv, index=False)
        print(f"[OUT] Refined grid results → {refined_csv}")
        print(f"\n  Refined top 5 (lowest S):")
        print(refined.head(5)[["fi","fe","f2","alpha","S"]].to_string(index=False))
        best = refined.iloc[0]
    else:
        best = coarse_best

    print(interpret(best, params))

    # --- Plots ---
    plot_S_surface(results, best["alpha"], output_dir, label=subfolder)
    plot_pdf_comparison(df, params, bins, obs_counts,
                        best["f1"], best["f2"], best["alpha"],
                        output_dir=output_dir,
                        label=subfolder)

    print("[DONE]\n")


if __name__ == "__main__":
    main()

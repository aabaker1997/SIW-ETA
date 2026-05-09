# ---------------------------------------------
# Simple OLS & Partial Regression Plot (Robust) with tqdm
# + Modular Extensions
# ---------------------------------------------
import os
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, pearsonr
from statsmodels.graphics.regressionplots import plot_partregress
from tqdm import tqdm

# -----------------------------
# Output directory setup
# -----------------------------
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

subfolder = input("Enter subfolder name for this run (e.g. NV_g2024_Pres): ").strip()
if not subfolder:
    subfolder = "unnamed"
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, subfolder)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Plots will be saved to: {OUTPUT_DIR}")


def save_fig(fig, name):
    """Save figure to output dir with subfolder name prefix, then show."""
    filename = f"{subfolder}_{name}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"  Saved: {filename}")
    plt.show()


# -----------------------------
# 0. Module Config
# -----------------------------
RUN_BUCKETING       = True
RUN_POLY_FIT        = True
RUN_DIP_TEST        = True
RUN_RESIDUALS       = True
RUN_PARTIAL_R2      = True   # only fires if extra columns exist
RUN_SHPILKIN        = True
RUN_HETERO          = True

# Heteroscedasticity config
HETERO_QUANTILES        = 4     # number of quantile groups (4 = quartiles)
HETERO_VAR_RATIO_THRESH = 3.0   # flag if top/bottom quantile variance ratio exceeds this

# Shpilkin histogram config
SHPILKIN_BINS       = 50    # number of bins across [0, 1] vote share range
                             # increase for finer resolution, decrease for smoother

# Bucketing config
BUCKET_N            = 10     # initial number of buckets
BUCKET_MIN_PCT      = 0.02   # minimum fraction of sample per interior bucket before merging
BUCKET_EDGE_PCT     = 0.03   # merge edge buckets until each has at least this fraction of sample

# Polynomial config
POLY_R2_GAIN_THRESH = 0.05   # flag if quadratic adds this much r² over linear

# -----------------------------
# 1. Config
# -----------------------------
csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "template.csv")

# -----------------------------
# 2. Load CSV
# -----------------------------
df = pd.read_csv(csv_file)
print("Raw dataframe shape:", df.shape)

# -----------------------------
# 3. Clean column names
# -----------------------------
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(' ', '_')
      .str.replace('%', 'pct')
)

# Sanitize column names for patsy compatibility:
# strip remaining non-alphanumeric/underscore chars, prefix leading digits with 'v_'
def sanitize_col(name):
    name = re.sub(r'[^a-z0-9_]', '_', name)   # replace anything not word-safe
    if name[0].isdigit():
        name = 'v_' + name                      # patsy can't start a token with a digit
    return name

import re
df.columns = [sanitize_col(c) for c in df.columns]
print("Sanitized columns:", df.columns.tolist())

# -----------------------------
# 4. Infer variables by column order
# -----------------------------
main_predictor = df.columns[0]
outcome_var    = df.columns[1]
extra_cols     = [c for c in df.columns[2:] if c not in (main_predictor, outcome_var)]
print(f"Main predictor : {main_predictor}")
print(f"Outcome variable: {outcome_var}")
if extra_cols:
    print(f"Extra columns (for partial r²): {extra_cols}")

# -----------------------------
# 5. Force numeric conversion
# -----------------------------
for col in [main_predictor, outcome_var] + extra_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# -----------------------------
# 6. Drop missing values
# -----------------------------
before_drop = len(df)
df = df.dropna(subset=[main_predictor, outcome_var] + extra_cols)
after_drop = len(df)
print(f"Rows before dropna: {before_drop}")
print(f"Rows after dropna:  {after_drop}")
n = len(df)

# -----------------------------
# 7. Correlation
# -----------------------------
corr = df[[outcome_var, main_predictor]].corr().iloc[0, 1]
print(f"Correlation between {outcome_var} and {main_predictor}: {corr:.4f}")

# -----------------------------
# 8. Fit OLS
# -----------------------------
model = smf.ols(f"{outcome_var} ~ {main_predictor}", data=df).fit()
print("\nOLS Summary:")
print(model.summary())

# -----------------------------
# 9. Partial Regression Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(6, 5))
plot_partregress(
    endog=outcome_var,
    exog_i=main_predictor,
    exog_others=[],
    data=df,
    ax=ax,
    obs_labels=False
)
ax.set_title(f"Partial Regression Plot: {main_predictor} vs {outcome_var}")
plt.tight_layout()
save_fig(fig, "01_partial_regression")

# -----------------------------
# 10. Pearson r
# -----------------------------
r, p = pearsonr(df[main_predictor], df[outcome_var])
print(f"Pearson r = {r:.3f}, p-value = {p:.4g}")

# -----------------------------
# 11. KDE Heatmap with tqdm
# -----------------------------
x = df[main_predictor].values
y = df[outcome_var].values

x_mu, x_sd = x.mean(), x.std()
y_mu, y_sd = y.mean(), y.std()
xz = (x - x_mu) / x_sd
yz = (y - y_mu) / y_sd

values = np.vstack([xz, yz])
kde = gaussian_kde(values, bw_method=0.3)

nx, ny = 300, 300
X_MIN, X_MAX = x.min(), x.max()
Y_MIN, Y_MAX = y.min(), y.max()
xg = np.linspace(X_MIN, X_MAX, nx)
yg = np.linspace(Y_MIN, Y_MAX, ny)
X, Y = np.meshgrid(xg, yg)
Xz = (X - x_mu) / x_sd
Yz = (Y - y_mu) / y_sd

Z = np.zeros_like(X, dtype=float)
print("Evaluating KDE on grid...")
for i in tqdm(range(ny), desc="KDE rows"):
    grid_points = np.vstack([Xz[i, :], Yz[i, :]])
    Z[i, :] = kde(grid_points)

fig, ax = plt.subplots(figsize=(7, 5))
cf = ax.contourf(X, Y, Z, levels=40, cmap="turbo")
ax.set_xlabel(main_predictor)
ax.set_ylabel(outcome_var)
ax.set_title(f"KDE Heatmap: {main_predictor} vs {outcome_var}")
ax.grid(True, linestyle="--", alpha=0.3)
fig.colorbar(cf, ax=ax, label="Density")
fig.tight_layout()
save_fig(fig, "02_kde_heatmap")


# =====================================================
# MODULE 1: BUCKETING EXHIBIT
# =====================================================
if RUN_BUCKETING:
    print("\n--- Module 1: Bucketing Exhibit ---")

    def make_robust_buckets(series, n_buckets, min_pct, edge_pct, n_total):
        """
        Bin series into n_buckets equal-width bins, then:
        1. Merge interior buckets smaller than min_pct of sample into neighbors.
        2. Merge edge (lowest/highest) buckets upward/inward until each
           constitutes at least edge_pct of sample.
        Returns list of (label, mask) tuples.
        """
        lo, hi = series.min(), series.max()
        edges = np.linspace(lo, hi, n_buckets + 1)
        bins = []
        for i in range(n_buckets):
            mask = (series >= edges[i]) & (series < edges[i + 1] if i < n_buckets - 1 else series <= edges[i + 1])
            bins.append(mask)

        # Merge interior bins below min_pct
        min_count = int(np.ceil(min_pct * n_total))
        merged = True
        while merged and len(bins) > 2:
            merged = False
            for i in range(1, len(bins) - 1):
                if bins[i].sum() < min_count:
                    # merge into smaller neighbor
                    if bins[i - 1].sum() <= bins[i + 1].sum():
                        bins[i - 1] = bins[i - 1] | bins[i]
                        bins.pop(i)
                    else:
                        bins[i + 1] = bins[i + 1] | bins[i]
                        bins.pop(i)
                    merged = True
                    break

        # Merge edges until they each meet edge_pct
        edge_count = int(np.ceil(edge_pct * n_total))
        while len(bins) > 2 and bins[0].sum() < edge_count:
            bins[1] = bins[0] | bins[1]
            bins.pop(0)
        while len(bins) > 2 and bins[-1].sum() < edge_count:
            bins[-2] = bins[-2] | bins[-1]
            bins.pop(-1)

        # Build labels from actual data ranges in each bin
        labels = []
        for b in bins:
            lo_b = series[b].min()
            hi_b = series[b].max()
            labels.append(f"{lo_b:.2f}–{hi_b:.2f}\n(n={b.sum()})")

        return list(zip(labels, bins))

    buckets = make_robust_buckets(
        df[main_predictor], BUCKET_N, BUCKET_MIN_PCT, BUCKET_EDGE_PCT, n
    )

    bucket_means = []
    bucket_errs  = []
    bucket_labels = []
    for label, mask in buckets:
        vals = df.loc[mask, outcome_var]
        bucket_means.append(vals.mean())
        bucket_errs.append(vals.sem())
        bucket_labels.append(label)

    fig, ax = plt.subplots(figsize=(max(8, len(buckets) * 1.2), 5))
    ax.bar(range(len(buckets)), bucket_means, yerr=bucket_errs,
           capsize=4, color="steelblue", alpha=0.8, edgecolor="white")
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels(bucket_labels, fontsize=8)
    ax.set_xlabel(f"{main_predictor} bucket")
    ax.set_ylabel(f"Mean {outcome_var} (± SEM)")
    ax.set_title(f"Bucketed Averages: {main_predictor} vs {outcome_var}\n"
                 f"(edge ≥{BUCKET_EDGE_PCT*100:.0f}% of sample, "
                 f"interior ≥{BUCKET_MIN_PCT*100:.0f}% of sample)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    save_fig(fig, "03_bucketed_averages")
    print(f"Rendered {len(buckets)} buckets after merging.")


# =====================================================
# MODULE 2: POLYNOMIAL VS LINEAR FIT
# =====================================================
if RUN_POLY_FIT:
    print("\n--- Module 2: Polynomial vs Linear Fit ---")

    x_vals = df[main_predictor].values
    y_vals = df[outcome_var].values

    # Linear
    coeffs1 = np.polyfit(x_vals, y_vals, 1)
    y_pred1  = np.polyval(coeffs1, x_vals)
    ss_res1  = np.sum((y_vals - y_pred1) ** 2)
    ss_tot   = np.sum((y_vals - y_vals.mean()) ** 2)
    r2_lin   = 1 - ss_res1 / ss_tot

    # Quadratic
    coeffs2 = np.polyfit(x_vals, y_vals, 2)
    y_pred2  = np.polyval(coeffs2, x_vals)
    ss_res2  = np.sum((y_vals - y_pred2) ** 2)
    r2_quad  = 1 - ss_res2 / ss_tot

    gain = r2_quad - r2_lin
    flagged = gain >= POLY_R2_GAIN_THRESH

    print(f"Linear r²:    {r2_lin:.4f}")
    print(f"Quadratic r²: {r2_quad:.4f}")
    print(f"r² gain:      {gain:.4f} {'⚠ NON-LINEAR SHAPE FLAGGED' if flagged else '(linear adequate)'}")

    x_plot = np.linspace(x_vals.min(), x_vals.max(), 300)
    y_lin  = np.polyval(coeffs1, x_plot)
    y_quad = np.polyval(coeffs2, x_plot)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x_vals, y_vals, alpha=0.3, s=15, color="gray", label="Data")
    ax.plot(x_plot, y_lin,  "b-",  linewidth=2, label=f"Linear r²={r2_lin:.3f}")
    ax.plot(x_plot, y_quad, "r--", linewidth=2, label=f"Quadratic r²={r2_quad:.3f}")
    ax.set_xlabel(main_predictor)
    ax.set_ylabel(outcome_var)
    ax.set_title(f"Linear vs Quadratic Fit\n{'⚠ Non-linear shape detected' if flagged else 'Linear adequate'}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "04_poly_vs_linear")


# =====================================================
# MODULE 3: DIP TEST FOR BIMODALITY
# =====================================================
if RUN_DIP_TEST:
    print("\n--- Module 3: Dip Test for Bimodality ---")
    try:
        from diptest import diptest
        dip, pval = diptest(df[outcome_var].values)
        print(f"Hartigan's dip statistic: {dip:.4f}")
        print(f"p-value: {pval:.4g}")
        if pval < 0.05:
            print("⚠ Significant bimodality detected in outcome variable (p < 0.05)")
        else:
            print("Outcome variable consistent with unimodal distribution (p ≥ 0.05)")

        # Also test predictor
        dip_x, pval_x = diptest(df[main_predictor].values)
        print(f"\nPredictor dip statistic: {dip_x:.4f}, p={pval_x:.4g}")
        if pval_x < 0.05:
            print("⚠ Significant bimodality detected in predictor variable")

    except ImportError:
        print("diptest package not installed. Run: pip install diptest")


# =====================================================
# MODULE 4: RESIDUAL PLOT
# =====================================================
if RUN_RESIDUALS:
    print("\n--- Module 4: Residual Plot ---")

    fitted   = model.fittedvalues
    residuals = model.resid

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Residuals vs fitted
    axes[0].scatter(fitted, residuals, alpha=0.3, s=15, color="steelblue")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Fitted values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Fitted\n(heteroscedasticity check)")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    # Residuals vs predictor (more useful for spotting non-linearity)
    axes[1].scatter(df[main_predictor], residuals, alpha=0.3, s=15, color="darkorange")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1)
    # Smooth trend through residuals
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(residuals, df[main_predictor], frac=0.3)
        axes[1].plot(smoothed[:, 0], smoothed[:, 1], "b-", linewidth=2, label="LOWESS")
        axes[1].legend()
    except Exception:
        pass
    axes[1].set_xlabel(main_predictor)
    axes[1].set_ylabel("Residuals")
    axes[1].set_title(f"Residuals vs {main_predictor}\n(non-linearity check)")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    save_fig(fig, "05_residuals")

    # Breusch-Pagan heteroscedasticity test
    try:
        from statsmodels.stats.diagnostic import het_breuschpagan
        bp_stat, bp_pval, _, _ = het_breuschpagan(residuals, model.model.exog)
        print(f"Breusch-Pagan test: stat={bp_stat:.4f}, p={bp_pval:.4g}")
        if bp_pval < 0.05:
            print("⚠ Heteroscedasticity detected — variance not constant across predictor range")
        else:
            print("No significant heteroscedasticity detected")
    except Exception as e:
        print(f"Breusch-Pagan test failed: {e}")


# =====================================================
# MODULE 5: PARTIAL R² DECOMPOSITION
# =====================================================
if RUN_PARTIAL_R2 and extra_cols:
    print("\n--- Module 5: Partial r² Decomposition ---")

    # Full model with all controls
    rhs_full = " + ".join([main_predictor] + extra_cols)
    model_full = smf.ols(f"{outcome_var} ~ {rhs_full}", data=df).fit()
    r2_full = model_full.rsquared
    print(f"Full model r² ({main_predictor} + controls): {r2_full:.4f}")

    # Controls-only model
    rhs_controls = " + ".join(extra_cols)
    model_controls = smf.ols(f"{outcome_var} ~ {rhs_controls}", data=df).fit()
    r2_controls = model_controls.rsquared
    print(f"Controls-only r²: {r2_controls:.4f}")

    # Partial r² of main predictor after controls
    partial_r2 = r2_full - r2_controls
    print(f"Partial r² of {main_predictor} after controls: {partial_r2:.4f}")
    print(f"(Controls explain {r2_controls/r2_full*100:.1f}% of the full model r²)")

    if partial_r2 < 0.10:
        print(f"⚠ {main_predictor} explains very little variance once controls are added")

    # Partial regression plot with controls
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_partregress(
        endog=outcome_var,
        exog_i=main_predictor,
        exog_others=extra_cols,
        data=df,
        ax=ax,
        obs_labels=False
    )
    ax.set_title(
        f"Partial Regression: {main_predictor} vs {outcome_var}\n"
        f"controlling for: {', '.join(extra_cols)}\n"
        f"partial r²={partial_r2:.4f}"
    )
    plt.tight_layout()
    save_fig(fig, "06_partial_r2")

    # Per-variable contribution table
    print("\nPer-variable r² contribution (sequential):")
    cumulative = 0.0
    for col in [main_predictor] + extra_cols:
        others = [c for c in [main_predictor] + extra_cols if c != col]
        rhs_without = " + ".join(others) if others else "1"
        try:
            m_without = smf.ols(f"{outcome_var} ~ {rhs_without}", data=df).fit()
            contribution = r2_full - m_without.rsquared
            print(f"  {col:<30} partial r²={contribution:.4f}")
        except Exception as e:
            print(f"  {col:<30} error: {e}")

elif RUN_PARTIAL_R2 and not extra_cols:
    print("\n--- Module 5: Partial r² ---")
    print("No extra columns found in CSV — skipping partial r² decomposition.")


# =====================================================
# MODULE 6: SHPILKIN HISTOGRAM
# Vote share distribution vs. fitted normal
# =====================================================
if RUN_SHPILKIN:
    print("\n--- Module 6: Shpilkin Histogram ---")
    from scipy.stats import norm

    y_vals = df[outcome_var].values
    mu     = y_vals.mean()
    sigma  = y_vals.std()

    # Bin edges across full [0,1] range so position is comparable across elections
    bin_edges  = np.linspace(0, 1, SHPILKIN_BINS + 1)
    bin_width  = bin_edges[1] - bin_edges[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    counts, _ = np.histogram(y_vals, bins=bin_edges)

    # Fitted normal scaled to match total count * bin_width
    # (i.e. same area under curve as the histogram)
    scale  = len(y_vals) * bin_width
    fitted = norm.pdf(bin_centers, loc=mu, scale=sigma) * scale

    # Residuals: observed - expected
    residuals_shp = counts - fitted
    excess_flag   = np.abs(residuals_shp) > (2 * np.sqrt(fitted + 1))  # ~2σ Poisson threshold

    print(f"Mean vote share : {mu:.4f}")
    print(f"Std dev         : {sigma:.4f}")
    print(f"Bins flagged >2σ from normal: {excess_flag.sum()} of {SHPILKIN_BINS}")

    # Bins near round numbers (every 5%) — classic Shpilkin fraud signal
    round_bins = [i for i, c in enumerate(bin_centers) if abs((c * 100) % 5) < (bin_width * 100 / 2)]
    if round_bins:
        round_excess = residuals_shp[round_bins]
        net_round_excess = round_excess[round_excess > 0].sum()
        print(f"Net excess count at round-number bins (multiples of 5%): {net_round_excess:.1f}")
        if net_round_excess > 10:
            print("⚠ Notable excess at round-number vote share values")

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(
        f"Shpilkin Histogram: {outcome_var}\n"
        f"mean={mu:.3f}, σ={sigma:.3f}, n={len(y_vals)}",
        fontsize=11, fontweight="bold"
    )

    # Top panel: raw counts vs fitted normal
    axes[0].bar(bin_centers, counts, width=bin_width * 0.9,
                color="steelblue", alpha=0.7, label="Observed count")
    axes[0].plot(bin_centers, fitted, "r-", linewidth=2, label="Fitted normal")
    # Mark flagged bins
    for i, flagged in enumerate(excess_flag):
        if flagged:
            axes[0].bar(bin_centers[i], counts[i], width=bin_width * 0.9,
                        color="red", alpha=0.5)
    axes[0].set_ylabel("Precinct count")
    axes[0].set_title("Observed distribution vs. fitted normal (red bars = >2σ excess)")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    # Bottom panel: residuals (observed - expected)
    bar_colors = ["red" if r > 0 else "steelblue" for r in residuals_shp]
    axes[1].bar(bin_centers, residuals_shp, width=bin_width * 0.9,
                color=bar_colors, alpha=0.7)
    axes[1].axhline(0, color="black", linewidth=0.8)
    # 2σ Poisson envelope
    envelope = 2 * np.sqrt(fitted + 1)
    axes[1].plot(bin_centers,  envelope, "k--", linewidth=1, alpha=0.5, label="±2σ envelope")
    axes[1].plot(bin_centers, -envelope, "k--", linewidth=1, alpha=0.5)
    axes[1].set_xlabel(f"{outcome_var} (vote share)")
    axes[1].set_ylabel("Observed − Expected")
    axes[1].set_title("Residuals from fitted normal")
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    # Shade round-number bins lightly in both panels
    for i in round_bins:
        for ax in axes:
            ax.axvspan(bin_edges[i], bin_edges[i + 1],
                       alpha=0.08, color="orange", zorder=0)

    plt.tight_layout()
    save_fig(fig, "07_shpilkin_histogram")


# =====================================================
# MODULE 7: HETEROSCEDASTICITY DETECTION
# Fan-shape / variance-by-quantile analysis
# =====================================================
if RUN_HETERO:
    print("\n--- Module 7: Heteroscedasticity Detection ---")
    from scipy.stats import levene, spearmanr

    resid_vals = model.resid.values
    pred_vals  = df[main_predictor].values

    # --- 7a. Spearman correlation between |residual| and predictor ---
    rho, rho_p = spearmanr(pred_vals, np.abs(resid_vals))
    print(f"Spearman r(|residual|, {main_predictor}): {rho:+.4f}  p={rho_p:.4g}")
    if abs(rho) > 0.2 and rho_p < 0.05:
        direction = "increases" if rho > 0 else "decreases"
        print(f"⚠ Variance {direction} with {main_predictor} (fan shape detected)")
    else:
        print("No significant fan shape detected")

    # --- 7b. Split into quantile groups, compute variance per group ---
    quantile_edges = np.quantile(pred_vals, np.linspace(0, 1, HETERO_QUANTILES + 1))
    quantile_edges = np.unique(quantile_edges)
    actual_q       = len(quantile_edges) - 1

    group_labels  = []
    group_vars    = []
    group_resids  = []

    for i in range(actual_q):
        lo = quantile_edges[i]
        hi = quantile_edges[i + 1]
        if i < actual_q - 1:
            mask = (pred_vals >= lo) & (pred_vals < hi)
        else:
            mask = (pred_vals >= lo) & (pred_vals <= hi)
        grp = resid_vals[mask]
        group_resids.append(grp)
        group_vars.append(np.var(grp, ddof=1))
        group_labels.append(f"{lo:.2f}–{hi:.2f}\n(n={mask.sum()})")

    # --- 7c. Variance ratio: top vs bottom quantile ---
    var_ratio = group_vars[-1] / group_vars[0] if group_vars[0] > 0 else float("nan")
    print(f"\nVariance by {main_predictor} quantile:")
    for lbl, v in zip(group_labels, group_vars):
        print(f"  {lbl.replace(chr(10), ' '):<30} var={v:.6f}  sd={v**0.5:.4f}")
    print(f"\nTop/bottom quantile variance ratio: {var_ratio:.2f}x")
    if var_ratio > HETERO_VAR_RATIO_THRESH:
        print(f"⚠ Variance ratio exceeds threshold ({HETERO_VAR_RATIO_THRESH}x) — strong fan shape")
    else:
        print(f"Variance ratio within threshold ({HETERO_VAR_RATIO_THRESH}x)")

    # --- 7d. Levene's test across quantile groups ---
    lev_stat, lev_p = (float("nan"), float("nan"))
    if actual_q >= 2:
        lev_stat, lev_p = levene(*group_resids)
        print(f"\nLevene's test across {actual_q} quantile groups: "
              f"stat={lev_stat:.4f}, p={lev_p:.4g}")
        if lev_p < 0.05:
            print("⚠ Levene's test significant — residual variance differs across predictor range")
        else:
            print("Levene's test not significant — variance roughly homogeneous")

    # --- 7e. Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    lev_p_str = f"{lev_p:.3g}" if not (lev_p != lev_p) else "n/a"  # nan check
    fig.suptitle(
        f"Heteroscedasticity Analysis: {main_predictor} vs {outcome_var}\n"
        f"Spearman ρ(|resid|, predictor)={rho:+.3f}  p={rho_p:.3g}  "
        f"Variance ratio={var_ratio:.2f}x  Levene p={lev_p_str}",
        fontsize=10, fontweight="bold"
    )

    # Left: variance by quantile group
    bar_colors = [
        "firebrick" if v == max(group_vars) else
        "steelblue" if v == min(group_vars) else
        "slategray"
        for v in group_vars
    ]
    axes[0].bar(range(actual_q), group_vars, color=bar_colors, alpha=0.8, edgecolor="white")
    axes[0].set_xticks(range(actual_q))
    axes[0].set_xticklabels(group_labels, fontsize=8)
    axes[0].set_ylabel("Residual variance")
    axes[0].set_xlabel(f"{main_predictor} quantile group")
    axes[0].set_title(f"Residual variance by {main_predictor} quantile\n(red=max, blue=min)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].annotate(
        f"ratio {var_ratio:.1f}x",
        xy=(actual_q - 1, group_vars[-1]),
        xytext=(-20, 8), textcoords="offset points",
        fontsize=9, color="firebrick", fontweight="bold"
    )

    # Right: scatter of residuals vs predictor with rolling SD envelope
    axes[1].scatter(pred_vals, resid_vals, alpha=0.25, s=12, color="darkorange")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1)

    sort_idx     = np.argsort(pred_vals)
    pred_sorted  = pred_vals[sort_idx]
    resid_sorted = resid_vals[sort_idx]
    window       = max(20, len(pred_vals) // 20)
    roll_sd      = pd.Series(resid_sorted).rolling(window, center=True, min_periods=5).std().values

    axes[1].plot(pred_sorted,  2 * roll_sd, "b-", linewidth=1.5, alpha=0.7, label="±2 rolling SD")
    axes[1].plot(pred_sorted, -2 * roll_sd, "b-", linewidth=1.5, alpha=0.7)
    axes[1].fill_between(pred_sorted, -2 * roll_sd, 2 * roll_sd, alpha=0.08, color="blue")

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(resid_vals, pred_vals, frac=0.3)
        axes[1].plot(smoothed[:, 0], smoothed[:, 1], "k-", linewidth=2, label="LOWESS")
    except Exception:
        pass

    axes[1].set_xlabel(main_predictor)
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residuals vs predictor with rolling ±2SD envelope")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    save_fig(fig, "08_heteroscedasticity")
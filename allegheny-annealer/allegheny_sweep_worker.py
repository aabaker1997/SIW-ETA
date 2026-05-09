"""
Allegheny County — Parallel Sweep Worker
=========================================
Runs a fast unconstrained random sweep and saves the best positive/negative
groupings as pickles for later aggregation.

Usage — launch multiple instances with different WORKER_ID values:
    python allegheny_sweep_worker.py 0
    python allegheny_sweep_worker.py 1
    python allegheny_sweep_worker.py 2
    ... etc

Each worker writes to its own subdirectory so there are no file collisions.
Run allegheny_sweep_aggregator.py afterward to pick the best across all workers.
"""

import sys
import os
import pickle
import geopandas as gpd
import networkx as nx
from libpysal.weights import Queen, Rook
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
WORKER_ID      = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N_SAMPLES      = 500_000   # per worker — adjust as desired

DATA_DIR   = r"os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")"
INPUT_FILE = os.path.join(DATA_DIR, "allegheny_results.csv")
SHP_DIR    = r"os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "shapefile")"
BASE_OUT   = r"os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "workers")"

OUTPUT_DIR = os.path.join(BASE_OUT, f"worker_{WORKER_ID:02d}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Each worker gets a different RNG seed so they explore different territory
RNG = np.random.default_rng(42 + WORKER_ID * 1_000_003)

print(f"Worker {WORKER_ID} — {N_SAMPLES:,} samples — output: {OUTPUT_DIR}")

# ── Municipality config ───────────────────────────────────────────────────────
MULTI_PRECINCT_MUNIS = sorted({
    "Aspinwall", "Avalon", "Baldwin Br", "Baldwin Tp", "Bellevue", "Ben Avon",
    "Bethel Park", "Brackenridge", "Braddock Hl", "Braddock", "Brentwood",
    "Bridgeville", "Carnegie", "Casl Shannon", "Cheswick", "Churchill",
    "Clairton", "Collier", "Corapolis", "Crafton", "Crescent", "Dormont",
    "Dravosburg", "Duquesne", "East Deer", "E McKeesport", "E Pittsburgh",
    "Edgewood", "Elizabeth Tp", "Emsworth", "Etna", "Fawn", "Findlay",
    "Forest Hills", "Forward", "Fox Chapel", "Franklin Pk", "Glassport",
    "Green Tree", "Hampton", "Harmar", "Harrison", "Homestead", "Indiana",
    "Ingram", "Jefferson Hl", "Kennedy", "Leet", "Liberty", "Marshall",
    "McCandless", "McKeesport", "McKees Rocks", "Millvale", "Monroeville",
    "Moon", "Mt Lebanon", "Mt Oliver", "Munhall", "Neville", "N Braddock",
    "N Fayette", "N Versailles", "Oakdale", "Oakmont", "O'Hara", "Ohio",
    "Penn Hills", "Pine", "Pitcairn", "Pittsburgh", "Plum", "Port Vue",
    "Rankin", "Reserve", "Richland", "Robinson", "Ross", "Scott", "Sewickley",
    "Shaler", "Sharpsburg", "S Fayette", "South Park", "Springdale", "Stowe",
    "Swissvale", "Tarentum", "Turtle Creek", "Up St Clair", "Verona",
    "Versailles", "West Deer", "W Homestead", "West Mifflin", "West View",
    "Whitaker", "Whitehall", "White Oak", "Wilkinsburg", "Wilkins",
    "Wilmerding",
}, key=len, reverse=True)

SINGLETON_OVERRIDES = {"Rosslyn Farms"}

MANUAL_MAP = {
    "springdal br": "springdale br",
    "ohara":        "o'hara",
}


def _normalise_shp_key(raw, csv_lower_map):
    s = raw.strip(); sl = s.lower()
    for prefix, repl in MANUAL_MAP.items():
        if sl.startswith(prefix):
            sl = repl + sl[len(prefix):]
            break
    return csv_lower_map.get(sl, s)


def assign_muni(precinct):
    if precinct in SINGLETON_OVERRIDES:
        return precinct
    for muni in MULTI_PRECINCT_MUNIS:
        if precinct.startswith(muni):
            return muni
    return precinct


# ── Load CSV ──────────────────────────────────────────────────────────────────
print("Loading CSV…")
df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip()
df = df[df["TotalEDay"] > 0].copy().reset_index(drop=True)
df["Muni"] = df["Precinct"].apply(assign_muni)
df["_join_key"] = df["Precinct"].str.strip()
csv_lower_map = {k.lower(): k for k in df["_join_key"]}

# ── Load shapefile ────────────────────────────────────────────────────────────
print("Loading shapefile…")
gdf = gpd.read_file(SHP_DIR)
gdf["_join_key"] = gdf["Muni_War_1"].apply(
    lambda x: _normalise_shp_key(x, csv_lower_map))
gdf = gdf.merge(df[["_join_key", "Muni"]].drop_duplicates(),
                on="_join_key", how="inner").reset_index(drop=True)

gdf_to_df = {}
for gi, row in gdf.iterrows():
    m = df.index[df["_join_key"] == row["_join_key"]].tolist()
    if m:
        gdf_to_df[gi] = m[0]

# ── Build adjacency graph ─────────────────────────────────────────────────────
print("Building contiguity graph…")
try:
    w = Rook.from_dataframe(gdf, silence_warnings=True)
except Exception:
    w = Queen.from_dataframe(gdf, silence_warnings=True)

_adj = {i: set() for i in range(len(df))}
for gi, gjs in w.neighbors.items():
    if gi not in gdf_to_df:
        continue
    di = gdf_to_df[gi]
    mi = df.loc[di, "Muni"]
    for gj in gjs:
        if gj not in gdf_to_df:
            continue
        dj = gdf_to_df[gj]
        if df.loc[dj, "Muni"] == mi:
            _adj[di].add(dj)
            _adj[dj].add(di)

# ── Municipality indexing ─────────────────────────────────────────────────────
muni_counts  = df.groupby("Muni").size()
multi_munis  = muni_counts[muni_counts > 1].index.tolist()
single_munis = muni_counts[muni_counts == 1].index.tolist()
muni_indices = {m: df.index[df["Muni"] == m].tolist() for m in multi_munis}
singles_df   = df[df["Muni"].isin(single_munis)].copy()
singles_df["HarrisPct"] = singles_df["HarrisEDay"] / singles_df["TotalEDay"]

# ── Precompute fast-path arrays ───────────────────────────────────────────────
arr_total  = df["TotalEDay"].values.astype(np.float64)
arr_harris = df["HarrisEDay"].values.astype(np.float64)

muni_arr = {}
for muni in multi_munis:
    idx = np.array(muni_indices[muni], dtype=np.intp)
    muni_arr[muni] = (arr_total[idx], arr_harris[idx])

sing_mask  = df["Muni"].isin(single_munis).values
sing_total = arr_total[sing_mask]
sing_pct   = arr_harris[sing_mask] / sing_total

S_sing_x  = sing_total.sum()
S_sing_y  = sing_pct.sum()
S_sing_xx = (sing_total * sing_total).sum()
S_sing_xy = (sing_total * sing_pct).sum()
S_sing_yy = (sing_pct   * sing_pct).sum()
N_sing    = float(len(sing_total))

_local_pos = {
    muni: {v: i for i, v in enumerate(muni_indices[muni])}
    for muni in multi_munis
}


def signed_r2_from_sums(Sx, Sy, Sxx, Sxy, Syy, n):
    if n < 2: return 0.0
    xb = Sx/n; yb = Sy/n
    cov  = Sxy/n - xb*yb
    varx = Sxx/n - xb*xb
    vary = Syy/n - yb*yb
    d = varx * vary
    if d <= 0: return 0.0
    return float(np.sign(cov) * cov * cov / d)


def _fast_sr2(grouping):
    Sx=S_sing_x; Sy=S_sing_y; Sxx=S_sing_xx; Sxy=S_sing_xy; Syy=S_sing_yy
    n=N_sing
    for muni, groups in grouping.items():
        t_arr, h_arr = muni_arr[muni]
        lp = _local_pos[muni]
        for g in groups:
            if len(g) == 1:
                lp0=lp[g[0]]; total=t_arr[lp0]; harris=h_arr[lp0]
            else:
                lpl=[lp[i] for i in g]; total=t_arr[lpl].sum(); harris=h_arr[lpl].sum()
            if total <= 0: continue
            pct=harris/total
            Sx+=total; Sy+=pct; Sxx+=total*total; Sxy+=total*pct; Syy+=pct*pct; n+=1.0
    return signed_r2_from_sums(Sx, Sy, Sxx, Sxy, Syy, n)


# ── Fast unconstrained partition ──────────────────────────────────────────────
def _partition_fast(muni):
    indices  = muni_indices[muni]
    n        = len(indices)
    n_groups = int(RNG.integers(1, n + 1))
    perm     = list(indices)
    RNG.shuffle(perm)
    groups   = [[perm[i]] for i in range(n_groups)]
    for idx in perm[n_groups:]:
        groups[int(RNG.integers(0, n_groups))].append(idx)
    return groups


def random_county_grouping_fast():
    return {muni: _partition_fast(muni) for muni in multi_munis}


# ── Sweep ─────────────────────────────────────────────────────────────────────
print(f"Sweeping {N_SAMPLES:,} samples…")
best_pos = {"signed_r2": -999, "grouping": None}
best_neg = {"signed_r2":  999, "grouping": None}
all_sr2  = []

with tqdm(total=N_SAMPLES, unit="sample", ncols=80) as pbar:
    for _ in range(N_SAMPLES):
        g   = random_county_grouping_fast()
        sr2 = _fast_sr2(g)
        all_sr2.append(sr2)
        if sr2 > best_pos["signed_r2"]:
            best_pos = {"signed_r2": sr2, "grouping": g}
        if sr2 < best_neg["signed_r2"]:
            best_neg = {"signed_r2": sr2, "grouping": g}
        pbar.update(1)

print(f"  Best positive: {best_pos['signed_r2']:+.4f}")
print(f"  Best negative: {best_neg['signed_r2']:+.4f}")
print(f"  Mean: {np.mean(all_sr2):+.4f}  Std: {np.std(all_sr2):.4f}")

# ── Save results ──────────────────────────────────────────────────────────────
results = {
    "worker_id":  WORKER_ID,
    "n_samples":  N_SAMPLES,
    "best_pos":   best_pos,
    "best_neg":   best_neg,
    "mean_sr2":   float(np.mean(all_sr2)),
    "std_sr2":    float(np.std(all_sr2)),
    "all_sr2":    all_sr2,
}
out_path = os.path.join(OUTPUT_DIR, "sweep_results.pkl")
with open(out_path, "wb") as f:
    pickle.dump(results, f)

# Also save a lightweight summary CSV (no grouping, just the sr2 values)
pd.DataFrame({"signed_r2": all_sr2}).to_csv(
    os.path.join(OUTPUT_DIR, "sweep_sr2.csv"), index=False)

print(f"Results saved to {out_path}")
print("Done.")
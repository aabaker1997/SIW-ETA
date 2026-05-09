"""
Allegheny County, PA 2024 Presidential Race — Turnout Correlation Script
(Municipality-Bounded, Contiguity-Constrained Version)

Same as allegheny_correlation_demo.py but the x-axis is EDay Turnout %
(TotalEDay / RegisteredVoters) instead of TotalEDay.

Question: can municipal-bounded contiguous re-aggregation shift the
correlation between EDay turnout rate and Harris vote share as dramatically
as it shifts the raw total-votes correlation?
"""

import geopandas as gpd
import networkx as nx
from libpysal.weights import Rook, Queen
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import math, time, os, warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = r"os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")"
INPUT_FILE = os.path.join(DATA_DIR, "allegheny_results.csv")
SHP_DIR    = r"os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "shapefile")"
OUTPUT_DIR = r"os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")\Turnout\contig"
SHP_JOIN_COL = "Muni_War_1"

os.makedirs(OUTPUT_DIR, exist_ok=True)

RNG = np.random.default_rng(42)
N_RANDOM_SAMPLES  = 500_000

SA_STEPS            = 25_000_000
SA_REPORT_INTERVAL  =  1_000_000
SA_STAGNATION_LIMIT =  2_000_000
PER_MUNI_ANNEALING  = False

T_END              = 1e-6
SA_STAGES = [
    (0.40, 1.0),
    (0.30, 0.1),
    (0.20, 0.01),
    (0.10, 1e-5),
]
N_CALIBRATION      = 2_000
TARGET_ACCEPT_RATE = 0.60

# ── Municipality lists ────────────────────────────────────────────────────────
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
df = df[df["RegisteredVoters"] > 0].copy().reset_index(drop=True)
df["Muni"]      = df["Precinct"].apply(assign_muni)
df["_join_key"] = df["Precinct"].str.strip()
print(f"  {len(df)} precincts loaded")

csv_lower_map = {k.lower(): k for k in df["_join_key"]}

# ── Load shapefile ────────────────────────────────────────────────────────────
print("Loading shapefile and joining…")
gdf = gpd.read_file(SHP_DIR)
gdf["_join_key"] = gdf[SHP_JOIN_COL].apply(
    lambda x: _normalise_shp_key(x, csv_lower_map))
n_before = len(gdf)
gdf = gdf.merge(df[["_join_key", "Muni"]].drop_duplicates(),
                on="_join_key", how="inner").reset_index(drop=True)
print(f"  {n_before} → {len(gdf)} rows after join")

gdf_to_df = {}
for gi, row in gdf.iterrows():
    m = df.index[df["_join_key"] == row["_join_key"]].tolist()
    if m:
        gdf_to_df[gi] = m[0]

print("Building Rook contiguity graph…")
try:
    w = Rook.from_dataframe(gdf, silence_warnings=True)
except Exception:
    w = Queen.from_dataframe(gdf, silence_warnings=True)

_df_neighbours_full = {i: set() for i in range(len(df))}
for gi, gjs in w.neighbors.items():
    if gi not in gdf_to_df: continue
    di = gdf_to_df[gi]
    for gj in gjs:
        if gj not in gdf_to_df: continue
        _df_neighbours_full[di].add(gdf_to_df[gj])

_adj = {}
for di in range(len(df)):
    muni_i = df.loc[di, "Muni"]
    _adj[di] = frozenset(
        j for j in _df_neighbours_full[di]
        if df.loc[j, "Muni"] == muni_i
    )
print("  Done.")

# ── Municipality indexing ─────────────────────────────────────────────────────
muni_counts  = df.groupby("Muni").size()
multi_munis  = muni_counts[muni_counts > 1].index.tolist()
single_munis = muni_counts[muni_counts == 1].index.tolist()
muni_indices = {m: df.index[df["Muni"] == m].tolist() for m in multi_munis}

muni_domains = {}
for muni in multi_munis:
    idx_set = set(df.index[df["Muni"] == muni].tolist())
    G = nx.Graph()
    G.add_nodes_from(idx_set)
    for i in idx_set:
        for j in _adj[i]:
            G.add_edge(i, j)
    muni_domains[muni] = [sorted(c) for c in nx.connected_components(G)]

print(f"  {len(multi_munis)} multi-precinct municipalities")
print()

# ── Vote arrays ───────────────────────────────────────────────────────────────
# x: group EDay turnout % = sum(TotalEDay) / sum(RegisteredVoters)
# y: group Harris EDay %  = sum(HarrisEDay) / sum(TotalEDay)
arr_eday       = df["TotalEDay"].values.astype(np.float64)
arr_registered = df["RegisteredVoters"].values.astype(np.float64)
arr_harris     = df["HarrisEDay"].values.astype(np.float64)

# Precompute per-muni local arrays
muni_arr = {}
for muni in multi_munis:
    idx = np.array(muni_indices[muni], dtype=np.intp)
    muni_arr[muni] = (arr_eday[idx], arr_registered[idx], arr_harris[idx])

_local_pos = {
    muni: {v: i for i, v in enumerate(muni_indices[muni])}
    for muni in multi_munis
}

# Singleton precomputed sums
sing_mask    = df["Muni"].isin(single_munis).values
sing_eday    = arr_eday[sing_mask]
sing_reg     = arr_registered[sing_mask]
sing_harris  = arr_harris[sing_mask]
sing_turnout = sing_eday / sing_reg          # x per singleton
sing_harrisp = sing_harris / sing_eday       # y per singleton

S_sing_x  = sing_turnout.sum()
S_sing_y  = sing_harrisp.sum()
S_sing_xx = (sing_turnout * sing_turnout).sum()
S_sing_xy = (sing_turnout * sing_harrisp).sum()
S_sing_yy = (sing_harrisp * sing_harrisp).sum()
N_sing    = float(len(sing_turnout))


# ── Articulation point prefilter ──────────────────────────────────────────────
def _find_aps(nodes):
    if len(nodes) <= 1: return set()
    node_set = set(nodes)
    disc = {}; low = {}; parent = {}; aps = set(); timer = [0]
    for root in nodes:
        if root in disc: continue
        disc[root] = low[root] = timer[0]; timer[0] += 1
        parent[root] = None; n_ch = {root: 0}
        stack = [(root, iter(v for v in _adj[root] if v in node_set))]
        while stack:
            u, ch = stack[-1]
            try:
                v = next(ch)
                if v not in disc:
                    disc[v] = low[v] = timer[0]; timer[0] += 1
                    parent[v] = u; n_ch[u] = n_ch.get(u, 0) + 1; n_ch[v] = 0
                    stack.append((v, iter(w for w in _adj[v] if w in node_set)))
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])
            except StopIteration:
                stack.pop()
                if stack:
                    p = parent[u]; low[p] = min(low[p], low[u])
                    if parent[p] is None:
                        if n_ch.get(p, 0) > 1: aps.add(p)
                    elif low[u] >= disc[p]:
                        aps.add(p)
    return aps

print("Precomputing articulation points…")
_is_ap = {}
for muni in multi_munis:
    for domain in muni_domains[muni]:
        aps = _find_aps(domain)
        for node in domain:
            _is_ap[node] = node in aps
print("  Done.")


# ── BFS and removability ──────────────────────────────────────────────────────
def _bfs_connected(node_set, start):
    if len(node_set) <= 1: return True
    visited = {start}; stack = [start]
    while stack:
        u = stack.pop()
        for v in _adj[u]:
            if v in node_set and v not in visited:
                visited.add(v); stack.append(v)
    return len(visited) == len(node_set)


def _removable(group_set, node):
    remaining = group_set - {node}
    if not remaining: return False
    nb_in = _adj[node] & group_set
    n = len(nb_in)
    if n == 0: return False
    if n == 1: return True
    if not _is_ap.get(node, True): return True
    return _bfs_connected(remaining, next(iter(nb_in)))


def _nodes_adj_between(set_a, set_b):
    for n in set_a:
        if _adj[n] & set_b: return True
    return False


# ── Signed r² ─────────────────────────────────────────────────────────────────
def signed_r2_from_sums(Sx, Sy, Sxx, Sxy, Syy, n):
    if n < 2: return 0.0
    xb = Sx / n; yb = Sy / n
    cov  = Sxy / n - xb * yb
    varx = Sxx / n - xb * xb
    vary = Syy / n - yb * yb
    d = varx * vary
    if d <= 0: return 0.0
    return float(np.sign(cov) * cov * cov / d)


def _fast_sr2(grouping):
    """Signed r² between group EDay turnout% and group HarrisPct."""
    Sx = S_sing_x; Sy = S_sing_y
    Sxx = S_sing_xx; Sxy = S_sing_xy; Syy = S_sing_yy; n = N_sing
    for muni, groups in grouping.items():
        e_arr, r_arr, h_arr = muni_arr[muni]
        lp = _local_pos[muni]
        for g in groups:
            lpos = [lp[i] for i in g]
            e = e_arr[lpos].sum(); r = r_arr[lpos].sum(); h = h_arr[lpos].sum()
            if e <= 0 or r <= 0: continue
            x = e / r; y = h / e
            Sx += x; Sy += y; Sxx += x*x; Sxy += x*y; Syy += y*y; n += 1
    return signed_r2_from_sums(Sx, Sy, Sxx, Sxy, Syy, n)


# ── Build combined DataFrame for plotting ─────────────────────────────────────
def build_combined(muni_groups):
    rows = []
    for muni, groups in muni_groups.items():
        e_arr, r_arr, h_arr = muni_arr[muni]
        lp = _local_pos[muni]
        for g in groups:
            lpos = [lp[i] for i in g]
            e = e_arr[lpos].sum(); r = r_arr[lpos].sum(); h = h_arr[lpos].sum()
            rows.append({"TurnoutPct": e/r if r > 0 else np.nan,
                         "HarrisPct":  h/e if e > 0 else np.nan,
                         "TotalEDay":  e,   "Muni": muni})
    sing_df = df[df["Muni"].isin(single_munis)].copy()
    sing_df["TurnoutPct"] = sing_df["TotalEDay"] / sing_df["RegisteredVoters"]
    sing_df["HarrisPct"]  = sing_df["HarrisEDay"] / sing_df["TotalEDay"]
    sing_rows = sing_df[["TurnoutPct", "HarrisPct", "TotalEDay", "Muni"]].to_dict("records")
    return pd.DataFrame(rows + sing_rows).dropna()


# ── Perturbation ──────────────────────────────────────────────────────────────
def _perturb_grouping(grouping):
    muni = multi_munis[int(RNG.integers(len(multi_munis)))]
    groups = [list(g) for g in grouping[muni]]
    gsets  = [set(g) for g in groups]
    can_split = any(len(g) >= 2 for g in groups)
    can_merge = len(groups) >= 2
    ops = []
    if can_split: ops.append(0)
    if can_merge: ops.append(1)
    if can_split and can_merge: ops.append(2)
    RNG.shuffle(ops)
    for op in ops:
        if op == 0:
            eligible = [i for i, g in enumerate(groups) if len(g) >= 2]
            RNG.shuffle(eligible)
            for gi in eligible:
                gs = gsets[gi]
                removable = [nd for nd in groups[gi] if _removable(gs, nd)]
                if removable:
                    nd = removable[int(RNG.integers(len(removable)))]
                    ng = [list(g) for g in groups]
                    ng[gi] = [x for x in ng[gi] if x != nd]
                    ng.append([nd])
                    new_g = dict(grouping); new_g[muni] = ng
                    return new_g
        elif op == 1:
            adj_pairs = [(i, j) for i in range(len(groups))
                         for j in range(i+1, len(groups))
                         if _nodes_adj_between(gsets[i], gsets[j])]
            if adj_pairs:
                i, j = adj_pairs[int(RNG.integers(len(adj_pairs)))]
                ng = [list(g) for g in groups]
                ng[i] = ng[i] + ng[j]; ng.pop(j)
                new_g = dict(grouping); new_g[muni] = ng
                return new_g
        else:
            moves = []
            for si, g_src in enumerate(groups):
                if len(g_src) < 2: continue
                gs_src = gsets[si]
                for nd in g_src:
                    if not _removable(gs_src, nd): continue
                    for di, gs_dst in enumerate(gsets):
                        if di != si and _adj[nd] & gs_dst:
                            moves.append((nd, si, di))
            if moves:
                nd, si, di = moves[int(RNG.integers(len(moves)))]
                ng = [list(g) for g in groups]
                ng[si] = [x for x in ng[si] if x != nd]
                ng[di] = ng[di] + [nd]
                new_g = dict(grouping); new_g[muni] = ng
                return new_g
    return grouping


# ── Plot helper ───────────────────────────────────────────────────────────────
def plot_scenario(combined_df, title, ax=None, highlight_muni="Pittsburgh"):
    x = combined_df["TurnoutPct"].values
    y = combined_df["HarrisPct"].values
    is_hi = (combined_df["Muni"] == highlight_muni).values
    own_fig = ax is None
    if own_fig: fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(x[~is_hi], y[~is_hi], c="#4a90d9", alpha=0.45,
               edgecolors="white", linewidths=0.3, s=40, zorder=3, label="Other")
    ax.scatter(x[is_hi],  y[is_hi],  c="#e05252", alpha=0.80,
               edgecolors="white", linewidths=0.3, s=55, zorder=4,
               label=highlight_muni)
    slope, intercept, r, _, _ = stats.linregress(x, y)
    r2 = r ** 2
    x_line = np.linspace(x.min(), x.max(), 300)
    ax.plot(x_line, slope * x_line + intercept,
            color="#222", linewidth=1.5, linestyle="--", zorder=5)
    direction = "+" if slope > 0 else "−"
    ax.text(0.97, 0.05, f"r² = {r2:.3f}  ({direction}slope)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.85))
    ax.set_xlabel("EDay Turnout % (TotalEDay / RegisteredVoters)", fontsize=10)
    ax.set_ylabel("Harris EDay Vote Share", fontsize=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.set_facecolor("#f9f9f9")
    if own_fig: plt.tight_layout()
    return {"title": title, "r2": r2,
            "signed_r2": float(np.sign(slope) * r2), "slope": slope,
            "n_groups": len(combined_df)}


# ── Initial grouping (one group per domain) ───────────────────────────────────
def make_initial_grouping():
    g = {}
    for muni in multi_munis:
        g[muni] = [list(d) for d in muni_domains[muni]]
    return g


# ── Scenario 1: Baseline ──────────────────────────────────────────────────────
print("Scenario 1: Baseline (natural precinct grouping)…")
init_g   = make_initial_grouping()
init_df  = build_combined(init_g)
fig1, ax1 = plt.subplots(figsize=(9, 5.5))
r = plot_scenario(init_df,
    "Baseline: Natural Precinct Grouping\n"
    "(EDay Turnout% vs Harris%, municipality-grouped)",
    ax=ax1)
fig1.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, "plot_01_baseline.png"), dpi=150)
plt.close(fig1)
print(f"  signed r² = {r['signed_r2']:+.3f}  (n={r['n_groups']})\n")
results_summary = [r]
baseline_sr2 = r["signed_r2"]


# ── Random sweep ──────────────────────────────────────────────────────────────
print(f"Random sweep: {N_RANDOM_SAMPLES:,} samples…")
best_pos = {"signed_r2": -999, "grouping": None}
best_neg = {"signed_r2":  999, "grouping": None}
all_sr2  = []

g = make_initial_grouping()
with tqdm(total=N_RANDOM_SAMPLES, unit="sample", ncols=80) as pbar:
    for _ in range(N_RANDOM_SAMPLES):
        g   = _perturb_grouping(g)
        sr2 = _fast_sr2(g)
        all_sr2.append(sr2)
        if sr2 > best_pos["signed_r2"]:
            best_pos = {"signed_r2": sr2, "grouping": {k: [list(x) for x in v] for k, v in g.items()}}
        if sr2 < best_neg["signed_r2"]:
            best_neg = {"signed_r2": sr2, "grouping": {k: [list(x) for x in v] for k, v in g.items()}}
        pbar.update(1)

print(f"\nSweep best pos: {best_pos['signed_r2']:+.3f}")
print(f"Sweep best neg: {best_neg['signed_r2']:+.3f}")
print(f"Mean: {np.mean(all_sr2):+.3f}  Std: {np.std(all_sr2):.3f}\n")

# Pre-annealing plots
for tag, bp in [("positive", best_pos), ("negative", best_neg)]:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plot_scenario(build_combined(bp["grouping"]),
                  f"Best {tag.title()} from Sweep  sr²={bp['signed_r2']:+.3f}",
                  ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"plot_02{'a' if tag=='positive' else 'b'}_sweep_{tag}.png"), dpi=150)
    plt.close(fig)
print("Pre-annealing scatter plots saved.")


# ── Annealing ─────────────────────────────────────────────────────────────────
def _calibrate_T(seed_g, maximize, label):
    deltas = []; g = seed_g
    for _ in range(N_CALIBRATION):
        cand  = _perturb_grouping(g)
        delta = _fast_sr2(cand) - _fast_sr2(g)
        sd    = delta if maximize else -delta
        if sd < 0: deltas.append(abs(sd))
        g = cand
    if not deltas: return 0.005
    T = -float(np.mean(deltas)) / math.log(TARGET_ACCEPT_RATE)
    T = max(T, 1e-6)
    print(f"  [{label}] T_START={T:.6f}  "
          f"(mean|Δ|={np.mean(deltas):.6f}, n_down={len(deltas)}/{N_CALIBRATION})")
    return T


def anneal(seed_g, target, label):
    maximize = (target == "pos")
    cur_g    = seed_g
    cur_sr2  = _fast_sr2(cur_g)
    best_g   = cur_g; best_sr2 = cur_sr2
    t0 = time.time(); step_total = 0; stagnant = 0

    print(f"  Starting {label} anneal  sr²={cur_sr2:+.4f}")
    T_START = _calibrate_T(seed_g, maximize, label)
    stage_temps = [(max(1, int(f * SA_STEPS)), T_START * m) for f, m in SA_STAGES]
    for i, (n, t) in enumerate(stage_temps):
        print(f"    Stage {i+1}: {n:>10,} steps  T_start={t:.6f}")

    for stage_idx, (stage_steps, stage_T) in enumerate(stage_temps):
        decay    = math.exp(math.log(T_END / stage_T) / stage_steps)
        T        = stage_T; n_acc = 0; n_imp = 0; stagnated = False
        print(f"  [{label}] Stage {stage_idx+1}: {stage_steps:,} steps  T_start={stage_T:.5f}")
        for _ in range(stage_steps):
            step_total += 1; stagnant += 1
            cand   = _perturb_grouping(cur_g)
            delta  = _fast_sr2(cand) - cur_sr2
            sd     = delta if maximize else -delta
            if sd >= 0 or math.log(RNG.random()) < sd / T:
                cur_g = cand; cur_sr2 += delta; n_acc += 1
                if sd > 0: n_imp += 1
                if (maximize and cur_sr2 > best_sr2) or \
                   (not maximize and cur_sr2 < best_sr2):
                    best_g = {k: [list(x) for x in v] for k, v in cur_g.items()}
                    best_sr2 = cur_sr2; stagnant = 0
            T *= decay
            if step_total % SA_REPORT_INTERVAL == 0:
                el  = time.time() - t0
                eta = (SA_STEPS - step_total) / (step_total / el)
                print(f"    step {step_total/1e6:5.1f}M  T={T:.2e}  "
                      f"accept={n_acc/SA_REPORT_INTERVAL:.2%}  "
                      f"improve={n_imp/SA_REPORT_INTERVAL:.2%}  "
                      f"stagnant={stagnant/1e6:.2f}M  "
                      f"cur={cur_sr2:+.4f}  best={best_sr2:+.4f}  "
                      f"eta={eta/3600:.2f}h")
                n_acc = 0; n_imp = 0
            if stagnant >= SA_STAGNATION_LIMIT:
                print(f"  [{label}] Stagnation at step {step_total:,} — best={best_sr2:+.4f}")
                stagnated = True; break
        if stagnated: break

    el = time.time() - t0
    print(f"  {label} done in {el/3600:.2f}h ({step_total:,} steps) — best={best_sr2:+.4f}")
    return {"signed_r2": best_sr2, "grouping": best_g}


print("Simulated annealing: positive target…")
sa_pos = anneal(best_pos["grouping"], "pos", "Positive")
print()
print("Simulated annealing: negative target…")
sa_neg = anneal(best_neg["grouping"], "neg", "Negative")
print()

print(f"Annealing best positive: {sa_pos['signed_r2']:+.3f}  (sweep was {best_pos['signed_r2']:+.3f})")
print(f"Annealing best negative: {sa_neg['signed_r2']:+.3f}  (sweep was {best_neg['signed_r2']:+.3f})")

# Post-annealing scatter plots
for tag, sa, sw in [("positive", sa_pos, best_pos), ("negative", sa_neg, best_neg)]:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    r = plot_scenario(build_combined(sa["grouping"]),
        f"Best {tag.title()} r² (muni-bounded, turnout x-axis)\n"
        f"Sweep: {sw['signed_r2']:+.3f} → Annealing: {sa['signed_r2']:+.3f}",
        ax=ax)
    fig.tight_layout()
    idx = "03" if tag == "positive" else "04"
    fig.savefig(os.path.join(OUTPUT_DIR, f"plot_{idx}_best_{tag}.png"), dpi=150)
    plt.close(fig)
    results_summary.append(r)

# Histograms
def _draw_hist(ax, data, title, xlim=None, vlines=None):
    plot_data = [v for v in data if xlim is None or xlim[0] <= v <= xlim[1]]
    ax.hist(plot_data, bins=80, color="#4a90d9", edgecolor="white", linewidth=0.4, alpha=0.85)
    ax.axvline(float(np.mean(data)), color="#222", linewidth=1.5, linestyle="--",
               label=f"Mean = {np.mean(data):+.3f}")
    ax.axvline(baseline_sr2, color="#e05252", linewidth=1.8, linestyle=":",
               label=f"Baseline = {baseline_sr2:+.3f}")
    if vlines:
        for (val, lbl), col in zip(vlines, ["#2ca02c","#ff7f0e","#9467bd","#8c564b"]):
            ax.axvline(val, color=col, linewidth=1.6, linestyle="-.", label=lbl)
    if xlim: ax.set_xlim(xlim)
    ax.set_xlabel("Signed r²", fontsize=10); ax.set_ylabel("Count", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left"); ax.set_facecolor("#f9f9f9")
    ax.grid(True, alpha=0.25, axis="y", linestyle=":")

vlines = [
    (best_pos["signed_r2"], f"Sweep pos = {best_pos['signed_r2']:+.3f}"),
    (best_neg["signed_r2"], f"Sweep neg = {best_neg['signed_r2']:+.3f}"),
    (sa_pos["signed_r2"],   f"SA pos = {sa_pos['signed_r2']:+.3f}"),
    (sa_neg["signed_r2"],   f"SA neg = {sa_neg['signed_r2']:+.3f}"),
]

fig_h1, ax_h1 = plt.subplots(figsize=(10, 5))
_draw_hist(ax_h1, all_sr2,
           f"Distribution of Signed r² — {N_RANDOM_SAMPLES:,} Muni-Bounded Turnout Groupings")
fig_h1.tight_layout()
fig_h1.savefig(os.path.join(OUTPUT_DIR, "plot_05a_distribution_natural.png"), dpi=150)
plt.close(fig_h1)

fig_h2, ax_h2 = plt.subplots(figsize=(10, 5))
_draw_hist(ax_h2, all_sr2,
           f"Distribution of Signed r² — {N_RANDOM_SAMPLES:,} Muni-Bounded Turnout Groupings\n"
           "x-axis bounded ±0.8",
           xlim=(-0.8, 0.8), vlines=vlines)
fig_h2.tight_layout()
fig_h2.savefig(os.path.join(OUTPUT_DIR, "plot_05b_distribution_bounded.png"), dpi=150)
plt.close(fig_h2)
print("Histograms saved.")

# Crosswalk CSVs
def make_crosswalk(grouping, filename):
    rows = []
    for muni, groups in grouping.items():
        e_arr, r_arr, h_arr = muni_arr[muni]
        lp = _local_pos[muni]
        for g in groups:
            lpos = [lp[i] for i in g]
            ge = e_arr[lpos].sum(); gr = r_arr[lpos].sum(); gh = h_arr[lpos].sum()
            gt_pct = ge / gr if gr > 0 else np.nan
            gh_pct = gh / ge if ge > 0 else np.nan
            grp_id = min(g)
            for nd in g:
                rows.append({
                    "SourcePrecinct":    df["Precinct"].iloc[nd],
                    "Muni":              muni,
                    "GroupID":           grp_id,
                    "GroupTurnoutPct":   round(gt_pct, 6),
                    "GroupHarrisPct":    round(gh_pct, 6),
                    "GroupTotalEDay":    int(ge),
                    "GroupRegistered":   int(gr),
                    "SourceTotalEDay":   int(arr_eday[nd]),
                    "SourceRegistered":  int(arr_registered[nd]),
                    "SourceHarrisEDay":  int(arr_harris[nd]),
                })
    # Add singletons
    for _, row in df[df["Muni"].isin(single_munis)].iterrows():
        nd = row.name
        ge = arr_eday[nd]; gr = arr_registered[nd]; gh = arr_harris[nd]
        rows.append({
            "SourcePrecinct":   row["Precinct"],
            "Muni":             row["Muni"],
            "GroupID":          nd,
            "GroupTurnoutPct":  round(ge/gr, 6) if gr > 0 else np.nan,
            "GroupHarrisPct":   round(gh/ge, 6) if ge > 0 else np.nan,
            "GroupTotalEDay":   int(ge),
            "GroupRegistered":  int(gr),
            "SourceTotalEDay":  int(ge),
            "SourceRegistered": int(gr),
            "SourceHarrisEDay": int(gh),
        })
    pd.DataFrame(rows).sort_values(
        ["Muni", "GroupID", "SourcePrecinct"]
    ).to_csv(os.path.join(OUTPUT_DIR, filename), index=False)

print("Writing crosswalk CSVs…")
make_crosswalk(best_pos["grouping"], "crosswalk_sweep_best_positive.csv")
make_crosswalk(best_neg["grouping"], "crosswalk_sweep_best_negative.csv")
make_crosswalk(sa_pos["grouping"],   "crosswalk_sa_best_positive.csv")
make_crosswalk(sa_neg["grouping"],   "crosswalk_sa_best_negative.csv")
print("Crosswalk CSVs saved.")

# Summary
summary_df = pd.DataFrame(results_summary)[["title","n_groups","r2","signed_r2","slope"]]
summary_df.to_csv(os.path.join(OUTPUT_DIR, "scenario_summary.csv"), index=False)
pd.DataFrame({"signed_r2": all_sr2}).to_csv(
    os.path.join(OUTPUT_DIR, "sweep_all_signed_r2.csv"), index=False)

print("\n── Scenario Summary ─────────────────────────────────────────────────")
print(summary_df.to_string(index=False))
print(f"\nAll files written to: {OUTPUT_DIR}")
print("Done.")
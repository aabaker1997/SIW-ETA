"""
Allegheny County, PA 2024 Presidential Race — Cross-Municipal Turnout Correlation Script
==========================================================================================
Identical to the cross-municipal script except the x-axis is EDay TURNOUT RATE
(TotalEDay / RegisteredVoters per group) instead of raw TotalEDay.

This tests whether the wide correlation range achievable with raw vote totals
also applies when using turnout percentage — a more normalised metric that
removes precinct size as a confound.

Expected result: narrower achievable range than the raw TotalEDay version,
since turnout % compresses the x-axis variance substantially.

Requires a RegisteredVoters column in the input CSV.
"""

import geopandas as gpd
import networkx as nx
from libpysal.weights import Queen, Rook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import math
import time
import os
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = r"os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")"
INPUT_FILE = os.path.join(DATA_DIR, "allegheny_results.csv")
SHP_DIR    = r"os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "shapefile")"
OUTPUT_DIR = r"os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "crossmuni")"
SHP_JOIN_COL = "Muni_War_1"

os.makedirs(OUTPUT_DIR, exist_ok=True)

RNG = np.random.default_rng(42)

N_RANDOM_SAMPLES   = 500_000
MIN_GROUPS         = 150     # lower bound on total groups
# MAX_GROUPS set dynamically to len(df) after load

# ── Annealing config ──────────────────────────────────────────────────────────
SA_STEPS            = 25_000_000
SA_REPORT_INTERVAL  =  1_000_000
SA_STAGNATION_LIMIT =  2_000_000

# ── Seed mode ─────────────────────────────────────────────────────────────────
# "full_split"       — start from every precinct its own group (guaranteed
#                      contiguous, annealer merges from 1327 down)
# "contiguous_sweep" — run N_CONTIG_SWEEP contiguous BFS samples first,
#                      use best as seed (slower startup, better seed quality)
# "from_pickle"      — load label arrays saved by a previous run
#                      (set PICKLE_POS / PICKLE_NEG to the .npy file paths)
SEED_MODE      = "full_split"
N_CONTIG_SWEEP = 2_000    # only used when SEED_MODE == "contiguous_sweep"
PICKLE_POS     = r"os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "crossmuni_turnout")\best_pos_label.npy"
PICKLE_NEG     = r"os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "crossmuni_turnout")\best_neg_label.npy"

T_END = 1e-6
# T_START calibrated per-run from actual perturbation deltas

# ── Name normalisation ────────────────────────────────────────────────────────
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
df["Muni"]      = df["Precinct"].apply(assign_muni)
df["_join_key"] = df["Precinct"].str.strip()
N = len(df)
MAX_GROUPS = N   # natural upper bound = every precinct its own group
csv_lower_map = {k.lower(): k for k in df["_join_key"]}
print(f"  {N} precincts loaded  (MIN_GROUPS={MIN_GROUPS}, MAX_GROUPS={MAX_GROUPS})")

# ── Load shapefile and build FULL (cross-municipal) adjacency ─────────────────
print("Loading shapefile…")
gdf = gpd.read_file(SHP_DIR)
gdf["_join_key"] = gdf[SHP_JOIN_COL].apply(
    lambda x: _normalise_shp_key(x, csv_lower_map))
gdf = gdf.merge(df[["_join_key"]].drop_duplicates(),
                on="_join_key", how="inner").reset_index(drop=True)

gdf_to_df = {}
for gi, row in gdf.iterrows():
    m = df.index[df["_join_key"] == row["_join_key"]].tolist()
    if m:
        gdf_to_df[gi] = m[0]

print("Building contiguity graph (county-wide, no municipal filter)…")
try:
    w = Queen.from_dataframe(gdf, silence_warnings=True)
except Exception:
    w = Rook.from_dataframe(gdf, silence_warnings=True)

# Full cross-municipal adjacency — no municipality filter applied
_adj = {i: set() for i in range(N)}
for gi, gjs in w.neighbors.items():
    if gi not in gdf_to_df:
        continue
    di = gdf_to_df[gi]
    for gj in gjs:
        if gj not in gdf_to_df:
            continue
        dj = gdf_to_df[gj]
        _adj[di].add(dj)
        _adj[dj].add(di)

# Convert to frozensets for immutable hot-path use
_adj = {i: frozenset(v) for i, v in _adj.items()}
print("  Done.")
print()

# ── Precompute vote arrays ────────────────────────────────────────────────────
# x-axis: group EDay turnout rate  = sum(TotalEDay) / sum(RegisteredVoters)
# y-axis: group Harris EDay pct    = sum(HarrisEDay) / sum(TotalEDay)
# _fast_sr2_label computes corr(x, y) where x = turnout rate, y = Harris pct
arr_total    = df["TotalEDay"].values.astype(np.float64)       # for Harris pct denom
arr_harris   = df["HarrisEDay"].values.astype(np.float64)      # Harris numerator
arr_reg      = df["RegisteredVoters"].values.astype(np.float64) # turnout denom

if "RegisteredVoters" not in df.columns:
    raise ValueError("RegisteredVoters column not found in CSV. "
                     "Please append registered voter counts before running.")


# ── Articulation point precomputation (county-wide graph) ─────────────────────
def _find_aps(nodes: list) -> set:
    """Iterative Tarjan AP algorithm on a subset of the county graph."""
    if len(nodes) <= 1:
        return set()
    node_set = set(nodes)
    disc = {}; low = {}; parent = {}; aps = set(); timer = [0]
    for root in nodes:
        if root in disc:
            continue
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


print("Precomputing articulation points (county-wide)…")
all_nodes = list(range(N))
county_aps = _find_aps(all_nodes)
_is_ap = np.zeros(N, dtype=bool)
for i in county_aps:
    _is_ap[i] = True
print(f"  {_is_ap.sum()} articulation points in county graph.")
print()


# ── BFS and removability ──────────────────────────────────────────────────────
def _bfs_connected(node_set: set, start: int) -> bool:
    if len(node_set) <= 1:
        return True
    visited = {start}; stack = [start]
    while stack:
        u = stack.pop()
        for v in _adj[u]:
            if v in node_set and v not in visited:
                visited.add(v); stack.append(v)
    return len(visited) == len(node_set)


def _removable(group_set: set, node: int) -> bool:
    remaining = group_set - {node}
    if not remaining:
        return False
    nb_in = _adj[node] & group_set
    n = len(nb_in)
    if n == 0: return False
    if n == 1: return True
    # NOTE: county-wide AP prefilter is NOT used here — a node may be a
    # non-AP of the full county graph but still be a cut vertex within a
    # specific merged group. Always BFS to be correct.
    return _bfs_connected(remaining, next(iter(nb_in)))


# ── Flat-label state representation ──────────────────────────────────────────
# label[i] = integer group id for precinct i
# Groups are identified by arbitrary integers; we compact periodically.
# boundary_nodes: set of precinct indices that have at least one neighbour
#                 with a different label — the only nodes eligible for moves.

def _init_label_full_split() -> np.ndarray:
    """Start state: every precinct its own group (label[i] = i)."""
    return np.arange(N, dtype=np.int32)


def _compact_labels(label: np.ndarray) -> np.ndarray:
    """Remap labels to 0..n_groups-1 with no gaps."""
    unique = np.unique(label)
    remap  = np.empty(label.max() + 1, dtype=np.int32)
    for new_id, old_id in enumerate(unique):
        remap[old_id] = new_id
    return remap[label]


def _build_boundary(label: np.ndarray) -> set:
    """Return set of precinct indices that border a different-labelled precinct."""
    boundary = set()
    for i in range(N):
        li = label[i]
        for j in _adj[i]:
            if label[j] != li:
                boundary.add(i)
                break
    return boundary


def _update_boundary(label: np.ndarray, boundary: set, node: int,
                     old_group: int, new_group: int) -> None:
    """
    Incrementally update boundary set after moving `node` from
    old_group → new_group. Only node and its neighbours can change status.
    Mutates boundary in place.
    """
    affected = {node} | (_adj[node])
    for i in affected:
        li = label[i]
        is_boundary = any(label[j] != li for j in _adj[i])
        if is_boundary:
            boundary.add(i)
        else:
            boundary.discard(i)


# ── Fast signed r² using np.bincount ─────────────────────────────────────────
def _fast_sr2_label(label: np.ndarray) -> float:
    """
    Compute signed r² directly from flat label array.
    x = group EDay turnout rate  (sum TotalEDay / sum RegisteredVoters)
    y = group Harris EDay pct    (sum HarrisEDay / sum TotalEDay)
    Uses np.bincount for O(N) vectorised group summation.
    """
    n_grps  = int(label.max()) + 1
    totals  = np.bincount(label, weights=arr_total,  minlength=n_grps)
    harris  = np.bincount(label, weights=arr_harris, minlength=n_grps)
    regs    = np.bincount(label, weights=arr_reg,    minlength=n_grps)
    mask    = (totals > 0) & (regs > 0)
    x       = totals[mask] / regs[mask]   # turnout rate
    y       = harris[mask] / totals[mask] # Harris pct

    n = float(len(x))
    if n < 2:
        return 0.0
    xb  = x.mean(); yb = y.mean()
    cov  = (x * y).mean() - xb * yb
    varx = (x * x).mean() - xb * xb
    vary = (y * y).mean() - yb * yb
    d = varx * vary
    if d <= 0:
        return 0.0
    return float(np.sign(cov) * cov * cov / d)


# ── Contiguity repair ────────────────────────────────────────────────────────
def _repair_contiguity(label: np.ndarray) -> np.ndarray:
    """
    Take an arbitrary (possibly non-contiguous) label array and return a new
    label array where every group is guaranteed contiguous.

    Method: for each existing group, find its connected components via BFS
    restricted to precincts with that label. The first component keeps the
    original label; each additional component gets a new unique label.

    The result may have more groups than the input (non-contiguous groups get
    split), but every group will be a single connected region. _compact_labels
    is called at the end to remove gaps in the label space.
    """
    new_label  = label.copy()
    next_id    = int(label.max()) + 1

    for grp in np.unique(label):
        members = list(np.where(label == grp)[0])
        if len(members) <= 1:
            continue

        member_set = set(members)
        visited    = set()

        # BFS to find connected components within this group
        first_component = True
        for start in members:
            if start in visited:
                continue
            # BFS from start, restricted to member_set
            component = {start}
            queue     = [start]
            visited.add(start)
            while queue:
                u = queue.pop()
                for v in _adj[u]:
                    if v in member_set and v not in visited:
                        visited.add(v)
                        component.add(v)
                        queue.append(v)

            if first_component:
                first_component = False
                # First component keeps original label — no change needed
            else:
                # Additional components get a new unique label
                for node in component:
                    new_label[node] = next_id
                next_id += 1

    return _compact_labels(new_label)


# ── Random contiguous partition for sweep ────────────────────────────────────
def _random_label_fast() -> np.ndarray:
    """Unconstrained — for sweep only. Random group assignment, no contiguity."""
    k = int(RNG.integers(MIN_GROUPS, MAX_GROUPS + 1))
    return RNG.integers(0, k, size=N, dtype=np.int32)


def _random_label() -> np.ndarray:
    """
    Generate a random contiguous partition of the county with
    MIN_GROUPS ≤ n_groups ≤ MAX_GROUPS via BFS region-growing from k seeds.
    """
    k = int(RNG.integers(MIN_GROUPS, MAX_GROUPS + 1))
    perm  = RNG.permutation(N)
    seeds = perm[:k]

    label    = np.full(N, -1, dtype=np.int32)
    frontier = {i: [int(seeds[i])] for i in range(k)}
    for i, s in enumerate(seeds):
        label[s] = i
    unassigned = set(perm[k:])

    while unassigned:
        progress = False
        for gid in range(k):
            for fn in list(frontier[gid]):
                for nb in _adj[fn]:
                    if label[nb] == -1:
                        label[nb] = gid
                        unassigned.discard(nb)
                        frontier[gid].append(nb)
                        progress = True
                        break
                if progress:
                    break
            if progress:
                break
        if not progress:
            # orphaned nodes — assign to nearest labelled neighbour
            for node in list(unassigned):
                for nb in _adj[node]:
                    if label[nb] != -1:
                        label[node] = label[nb]
                        unassigned.discard(node)
                        break
            if unassigned:
                # last resort
                for node in unassigned:
                    label[node] = 0
                unassigned.clear()

    return label


# ── Perturbation ──────────────────────────────────────────────────────────────
def _perturb_label(label: np.ndarray,
                   boundary: set,
                   n_groups: int) -> tuple:
    """
    Move one boundary precinct to an adjacent group, maintaining contiguity
    and group count bounds. Returns (new_label, new_boundary, new_n_groups).
    Returns originals unchanged if no valid move found.

    Moves:
      merge  — move node to neighbour's group (reduces n_groups if src empties)
      split  — node stays but gets a new unique group id (increases n_groups)
                only allowed if current n_groups < MAX_GROUPS
    """
    if not boundary:
        return label, boundary, n_groups

    # Pick a random boundary node
    bd_list = list(boundary)
    node    = int(bd_list[int(RNG.integers(len(bd_list)))])
    src_grp = int(label[node])

    # Build src group set (needed for removability check)
    src_members = set(np.where(label == src_grp)[0])

    # Decide available moves
    # merge: move node to an adjacent group
    # split: peel node into its own new group (if n_groups < MAX_GROUPS)
    adj_groups = {int(label[nb]) for nb in _adj[node] if label[nb] != src_grp}

    can_merge = (bool(adj_groups) and
                 (len(src_members) > 1 or n_groups > MIN_GROUPS))
    can_split = (n_groups < MAX_GROUPS and
                 len(src_members) > 1)

    ops = []
    if can_merge: ops.append("merge")
    if can_split: ops.append("split")
    if not ops:
        return label, boundary, n_groups

    op = ops[int(RNG.integers(len(ops)))]

    if op == "merge":
        # Check node is removable from src (so src stays connected after move)
        if len(src_members) > 1 and not _removable(src_members, node):
            return label, boundary, n_groups

        dst_grp  = int(RNG.choice(list(adj_groups)))
        new_label = label.copy()
        new_label[node] = dst_grp

        # If src is now empty, n_groups decreases
        new_n = n_groups - (1 if len(src_members) == 1 else 0)
        if new_n < MIN_GROUPS:
            return label, boundary, n_groups

        new_boundary = set(boundary)
        _update_boundary(new_label, new_boundary, node, src_grp, dst_grp)
        return new_label, new_boundary, new_n

    else:  # split
        if not _removable(src_members, node):
            return label, boundary, n_groups

        new_grp   = int(label.max()) + 1
        new_label = label.copy()
        new_label[node] = new_grp
        new_n = n_groups + 1

        new_boundary = set(boundary)
        _update_boundary(new_label, new_boundary, node, src_grp, new_grp)
        return new_label, new_boundary, new_n


# ── Plot helper ───────────────────────────────────────────────────────────────
def plot_scenario_label(label: np.ndarray, title: str,
                        ax=None, highlight_muni: str = "Pittsburgh") -> dict:
    """Build combined DataFrame from label array and plot."""
    n_grps  = int(label.max()) + 1
    totals  = np.bincount(label, weights=arr_total,  minlength=n_grps)
    harris  = np.bincount(label, weights=arr_harris, minlength=n_grps)
    regs    = np.bincount(label, weights=arr_reg,    minlength=n_grps)
    mask    = (totals > 0) & (regs > 0)
    x_all   = totals[mask] / regs[mask]   # turnout rate
    y_all   = harris[mask] / totals[mask] # Harris pct

    # For highlighting Pittsburgh: check if any precinct in each group is Pitt
    pitt_mask_precinct = (df["Muni"] == highlight_muni).values
    pitt_grps = set(label[pitt_mask_precinct].tolist())
    active_labels = np.where(mask)[0]
    is_hi = np.array([lbl in pitt_grps for lbl in active_labels])

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.scatter(x_all[~is_hi], y_all[~is_hi], c="#4a90d9", alpha=0.45,
               edgecolors="white", linewidths=0.3, s=40, zorder=3, label="Other")
    ax.scatter(x_all[is_hi],  y_all[is_hi],  c="#e05252", alpha=0.80,
               edgecolors="white", linewidths=0.3, s=55, zorder=4,
               label=highlight_muni)

    slope, intercept, r, _, _ = stats.linregress(x_all, y_all)
    r2     = r ** 2
    x_line = np.linspace(x_all.min(), x_all.max(), 300)
    ax.plot(x_line, slope * x_line + intercept,
            color="#222", linewidth=1.5, linestyle="--", zorder=5)

    direction = "+" if slope > 0 else "−"
    ax.text(0.97, 0.05,
            f"r² = {r2:.3f}  ({direction}slope)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.85))

    ax.set_xlabel("EDay Turnout Rate in Group (TotalEDay / RegisteredVoters)", fontsize=10)
    ax.set_ylabel("Harris EDay Vote Share", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.set_facecolor("#f9f9f9")

    if own_fig:
        plt.tight_layout()

    return {"title": title, "r2": r2,
            "signed_r2": float(np.sign(slope) * r2),
            "slope": slope,
            "n_groups": int(mask.sum()),
            "n_highlighted": int(is_hi.sum())}


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 1 — Baseline (every precinct its own group)
# ═══════════════════════════════════════════════════════════════════════════════
print("Scenario 1: Baseline (full granularity)…")
label_baseline = _init_label_full_split()

fig1, ax1 = plt.subplots(figsize=(9, 5.5))
r = plot_scenario_label(label_baseline,
                        "Baseline: All Precincts as Reported\n"
                        "(HarrisPct recalculated from raw totals, no re-aggregation)",
                        ax=ax1)
fig1.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, "plot_01_baseline.png"), dpi=150)
plt.close(fig1)
print(f"  signed r² = {r['signed_r2']:+.3f}  (n={r['n_groups']})\n")
results_summary = [r]
baseline_sr2 = r["signed_r2"]


# ═══════════════════════════════════════════════════════════════════════════════
# RANDOM SWEEP
# ═══════════════════════════════════════════════════════════════════════════════
print(f"Random sweep: {N_RANDOM_SAMPLES:,} cross-municipal groupings (unconstrained seeds)…")
print(f"  Group count bounds: {MIN_GROUPS} – {MAX_GROUPS}")
print()

best_pos = {"signed_r2": -999, "label": None}
best_neg = {"signed_r2":  999, "label": None}
all_sr2  = []

with tqdm(total=N_RANDOM_SAMPLES, unit="sample", ncols=80) as pbar:
    for _ in range(N_RANDOM_SAMPLES):
        lbl = _random_label_fast()   # unconstrained — fast seed search
        sr2 = _fast_sr2_label(lbl)
        all_sr2.append(sr2)
        if sr2 > best_pos["signed_r2"]:
            best_pos = {"signed_r2": sr2, "label": lbl.copy()}
        if sr2 < best_neg["signed_r2"]:
            best_neg = {"signed_r2": sr2, "label": lbl.copy()}
        pbar.update(1)

print()
print(f"Random sweep best positive: {best_pos['signed_r2']:+.3f}")
print(f"Random sweep best negative: {best_neg['signed_r2']:+.3f}")
print(f"Mean signed r²:             {np.mean(all_sr2):+.3f}")
print(f"Std:                        {np.std(all_sr2):.3f}")
print()

# Pre-annealing scatter plots
fig_sp, ax_sp = plt.subplots(figsize=(9, 5.5))
plot_scenario_label(best_pos["label"],
                    f"Best Positive r² from Sweep ({N_RANDOM_SAMPLES:,} samples)\n"
                    f"Signed r² = {best_pos['signed_r2']:+.3f}  (before annealing)",
                    ax=ax_sp)
fig_sp.tight_layout()
fig_sp.savefig(os.path.join(OUTPUT_DIR, "plot_02a_sweep_best_positive.png"), dpi=150)
plt.close(fig_sp)

fig_sn, ax_sn = plt.subplots(figsize=(9, 5.5))
plot_scenario_label(best_neg["label"],
                    f"Best Negative r² from Sweep ({N_RANDOM_SAMPLES:,} samples)\n"
                    f"Signed r² = {best_neg['signed_r2']:+.3f}  (before annealing)",
                    ax=ax_sn)
fig_sn.tight_layout()
fig_sn.savefig(os.path.join(OUTPUT_DIR, "plot_02b_sweep_best_negative.png"), dpi=150)
plt.close(fig_sn)
print("Pre-annealing scatter plots saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# SEED SELECTION
# ═══════════════════════════════════════════════════════════════════════════════
import numpy as _np_seed  # already imported as np — just for clarity below

print(f"Seed mode: {SEED_MODE}")

if SEED_MODE == "full_split":
    seed_pos = _init_label_full_split()
    seed_neg = _init_label_full_split()
    print(f"  Seeds: full split ({len(df)} groups each)")

elif SEED_MODE == "contiguous_sweep":
    print(f"  Running {N_CONTIG_SWEEP:,} contiguous BFS samples for seed…")
    cs_best_pos = {"signed_r2": -999, "label": None}
    cs_best_neg = {"signed_r2":  999, "label": None}
    with tqdm(total=N_CONTIG_SWEEP, unit="sample", ncols=80) as pbar:
        for _ in range(N_CONTIG_SWEEP):
            lbl = _random_label()
            sr2 = _fast_sr2_label(lbl)
            if sr2 > cs_best_pos["signed_r2"]:
                cs_best_pos = {"signed_r2": sr2, "label": lbl.copy()}
            if sr2 < cs_best_neg["signed_r2"]:
                cs_best_neg = {"signed_r2": sr2, "label": lbl.copy()}
            pbar.update(1)
    seed_pos = cs_best_pos["label"]
    seed_neg = cs_best_neg["label"]
    print(f"  Contiguous sweep best pos: {cs_best_pos['signed_r2']:+.4f}")
    print(f"  Contiguous sweep best neg: {cs_best_neg['signed_r2']:+.4f}")

elif SEED_MODE == "from_pickle":
    seed_pos = np.load(PICKLE_POS)
    seed_neg = np.load(PICKLE_NEG)
    print(f"  Loaded pos seed: {PICKLE_POS}  ({np.unique(seed_pos).size} groups)")
    print(f"  Loaded neg seed: {PICKLE_NEG}  ({np.unique(seed_neg).size} groups)")

else:
    raise ValueError(f"Unknown SEED_MODE: {SEED_MODE!r}. "
                     f"Must be 'full_split', 'contiguous_sweep', or 'from_pickle'.")

print()


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATED ANNEALING
# ═══════════════════════════════════════════════════════════════════════════════
T_END = 1e-6

SA_STAGES = [
    (0.40, 1.0),
    (0.30, 0.1),
    (0.20, 0.01),
    (0.10, 1e-5),
]

N_CALIBRATION      = 2_000
TARGET_ACCEPT_RATE = 0.60

def _calibrate_T_start_label(seed_label: np.ndarray,
                              seed_bd: set, seed_n: int,
                              maximize: bool, label_str: str) -> float:
    deltas = []
    lbl = seed_label.copy(); bd = set(seed_bd); n = seed_n
    for _ in range(N_CALIBRATION):
        new_lbl, new_bd, new_n = _perturb_label(lbl, bd, n)
        if new_lbl is not lbl:
            delta = _fast_sr2_label(new_lbl) - _fast_sr2_label(lbl)
            sd    = delta if maximize else -delta
            if sd < 0:
                deltas.append(abs(sd))
            lbl = new_lbl; bd = new_bd; n = new_n
    if not deltas:
        T = 0.005
    else:
        mean_d = float(np.mean(deltas))
        T = -mean_d / math.log(TARGET_ACCEPT_RATE)
        T = max(T, 1e-6)
    print(f"  [{label_str}] T_START calibrated to {T:.6f}  "
          f"(mean |delta|={np.mean(deltas) if deltas else 0:.6f}, "
          f"n_downhill={len(deltas)}/{N_CALIBRATION})")
    return T


def anneal_label(seed_label: np.ndarray, target: str, label_str: str) -> dict:
    maximize = (target == "pos")

    current_label = seed_label.copy()
    current_label = _compact_labels(current_label)
    current_label = _repair_contiguity(current_label)
    n_after_repair = int(np.unique(current_label).size)
    current_sr2   = _fast_sr2_label(current_label)
    current_n     = n_after_repair
    current_bd    = _build_boundary(current_label)

    best_label  = current_label.copy()
    best_sr2    = current_sr2

    t_start = time.time()
    step_total = 0
    steps_since_improve = 0

    print(f"  Starting {label_str} anneal  sr²={current_sr2:+.4f}  "
          f"n_groups={current_n}  (seed had {int(np.unique(seed_label).size)}, "
          f"repaired to {n_after_repair})")

    # Calibrate T_START from actual perturbation deltas at this seed
    T_START_cal = _calibrate_T_start_label(
        current_label, current_bd, current_n, maximize, label_str)

    stage_steps_temps = [
        (max(1, int(f * SA_STEPS)), T_START_cal * m)
        for f, m in SA_STAGES
    ]
    for i, (n, t) in enumerate(stage_steps_temps):
        print(f"    Stage {i+1}: {n:>10,} steps  T_start={t:.6f}")

    for stage_idx, (stage_steps, stage_T) in enumerate(stage_steps_temps):
        decay     = math.exp(math.log(T_END / stage_T) / stage_steps)
        T         = stage_T
        n_accept  = 0; n_improve = 0
        stagnated = False

        print(f"  [{label_str}] Stage {stage_idx+1}: "
              f"{stage_steps:,} steps  T_start={stage_T:.5f}")

        for _ in range(stage_steps):
            step_total          += 1
            steps_since_improve += 1

            new_label, new_bd, new_n = _perturb_label(
                current_label, current_bd, current_n)

            if new_label is current_label:
                T *= decay
                continue

            new_sr2      = _fast_sr2_label(new_label)
            delta        = new_sr2 - current_sr2
            signed_delta = delta if maximize else -delta

            if signed_delta >= 0:
                current_label = new_label
                current_sr2   = new_sr2
                current_bd    = new_bd
                current_n     = new_n
                n_accept += 1; n_improve += 1
                if (maximize and new_sr2 > best_sr2) or \
                   (not maximize and new_sr2 < best_sr2):
                    best_label  = new_label.copy()
                    best_sr2    = new_sr2
                    steps_since_improve = 0
            else:
                if math.log(RNG.random()) < signed_delta / T:
                    current_label = new_label
                    current_sr2   = new_sr2
                    current_bd    = new_bd
                    current_n     = new_n
                    n_accept += 1

            T *= decay

            if step_total % SA_REPORT_INTERVAL == 0:
                elapsed   = time.time() - t_start
                rate      = step_total / elapsed
                remaining = (SA_STEPS - step_total) / rate
                print(
                    f"    step {step_total/1e6:5.1f}M  T={T:.2e}  "
                    f"accept={n_accept/SA_REPORT_INTERVAL:.2%}  "
                    f"improve={n_improve/SA_REPORT_INTERVAL:.2%}  "
                    f"stagnant={steps_since_improve/1e6:.2f}M  "
                    f"n_grps={current_n}  "
                    f"cur={current_sr2:+.4f}  best={best_sr2:+.4f}  "
                    f"eta={remaining/3600:.2f}h"
                )
                n_accept = 0; n_improve = 0

            if steps_since_improve >= SA_STAGNATION_LIMIT:
                print(f"  [{label_str}] Stagnation at step {step_total:,} — "
                      f"best={best_sr2:+.4f}  Terminating early.")
                stagnated = True
                break

        if stagnated:
            break

    elapsed = time.time() - t_start
    print(f"  {label_str} done in {elapsed/3600:.2f}h ({step_total:,} steps) "
          f"— best={best_sr2:+.4f}  n_groups={np.unique(best_label).size}")
    return {"signed_r2": best_sr2, "label": best_label}


print("Simulated annealing: positive target…")
sa_pos = anneal_label(seed_pos, target="pos", label_str="Positive")
np.save(PICKLE_POS, sa_pos["label"])
print(f"  Best positive label saved to {PICKLE_POS}")
print()

print("Simulated annealing: negative target…")
sa_neg = anneal_label(seed_neg, target="neg", label_str="Negative")
np.save(PICKLE_NEG, sa_neg["label"])
print(f"  Best negative label saved to {PICKLE_NEG}")
print()

print(f"Annealing best positive: {sa_pos['signed_r2']:+.3f}  "
      f"(sweep was {best_pos['signed_r2']:+.3f})")
print(f"Annealing best negative: {sa_neg['signed_r2']:+.3f}  "
      f"(sweep was {best_neg['signed_r2']:+.3f})")
print()


# ── Post-annealing scatter plots ──────────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(9, 5.5))
r_pos = plot_scenario_label(
    sa_pos["label"],
    f"Best Positive r² (cross-municipal, sweep + annealing)\n"
    f"Sweep: {best_pos['signed_r2']:+.3f} → Annealing: {sa_pos['signed_r2']:+.3f}",
    ax=ax4)
fig4.tight_layout()
fig4.savefig(os.path.join(OUTPUT_DIR, "plot_03_best_positive_r2.png"), dpi=150)
plt.close(fig4)
results_summary.append(r_pos)

fig5, ax5 = plt.subplots(figsize=(9, 5.5))
r_neg = plot_scenario_label(
    sa_neg["label"],
    f"Best Negative r² (cross-municipal, sweep + annealing)\n"
    f"Sweep: {best_neg['signed_r2']:+.3f} → Annealing: {sa_neg['signed_r2']:+.3f}",
    ax=ax5)
fig5.tight_layout()
fig5.savefig(os.path.join(OUTPUT_DIR, "plot_04_best_negative_r2.png"), dpi=150)
plt.close(fig5)
results_summary.append(r_neg)


# ── Distribution histograms ───────────────────────────────────────────────────
def _draw_histogram(ax, data, title, x_lim=None, extra_vlines=None):
    plot_data = [v for v in data if x_lim is None or x_lim[0] <= v <= x_lim[1]]
    ax.hist(plot_data, bins=80, color="#4a90d9", edgecolor="white",
            linewidth=0.4, alpha=0.85)
    ax.axvline(float(np.mean(data)), color="#222", linewidth=1.5,
               linestyle="--", label=f"Mean = {np.mean(data):+.3f}")
    ax.axvline(baseline_sr2, color="#e05252", linewidth=1.8,
               linestyle=":", label=f"Baseline = {baseline_sr2:+.3f}")
    if extra_vlines:
        for (val, lbl), col in zip(extra_vlines,
                                   ["#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]):
            ax.axvline(val, color=col, linewidth=1.6, linestyle="-.", label=lbl)
    if x_lim:
        ax.set_xlim(x_lim)
    ax.set_xlabel("Signed r²", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_facecolor("#f9f9f9")
    ax.grid(True, alpha=0.25, axis="y", linestyle=":")

extra_vlines = [
    (best_pos["signed_r2"], f"Sweep best pos = {best_pos['signed_r2']:+.3f}"),
    (best_neg["signed_r2"], f"Sweep best neg = {best_neg['signed_r2']:+.3f}"),
    (sa_pos["signed_r2"],   f"SA best pos = {sa_pos['signed_r2']:+.3f}"),
    (sa_neg["signed_r2"],   f"SA best neg = {sa_neg['signed_r2']:+.3f}"),
]

fig_h1, ax_h1 = plt.subplots(figsize=(10, 5))
_draw_histogram(ax_h1, all_sr2,
                f"Distribution of Signed r² — {N_RANDOM_SAMPLES:,} Cross-Municipal Groupings\n"
                "Natural x-axis")
plt.tight_layout()
fig_h1.savefig(os.path.join(OUTPUT_DIR, "plot_05a_distribution_natural.png"), dpi=150)
plt.close(fig_h1)

fig_h2, ax_h2 = plt.subplots(figsize=(10, 5))
_draw_histogram(ax_h2, all_sr2,
                f"Distribution of Signed r² — {N_RANDOM_SAMPLES:,} Cross-Municipal Groupings\n"
                "x-axis bounded ±0.8",
                x_lim=(-0.8, 0.8), extra_vlines=extra_vlines)
plt.tight_layout()
fig_h2.savefig(os.path.join(OUTPUT_DIR, "plot_05b_distribution_bounded.png"), dpi=150)
plt.close(fig_h2)
print("Distribution histograms saved.")


# ── Crosswalk CSV ─────────────────────────────────────────────────────────────
def make_crosswalk_label(label: np.ndarray, filename: str) -> None:
    """One row per source precinct showing its group assignment and group totals."""
    n_grps  = int(label.max()) + 1
    totals  = np.bincount(label, weights=arr_total,  minlength=n_grps)
    harris  = np.bincount(label, weights=arr_harris, minlength=n_grps)
    regs    = np.bincount(label, weights=arr_reg,    minlength=n_grps)
    rows = []
    for i in range(N):
        grp = int(label[i])
        gt  = totals[grp]; gh = harris[grp]; gr = regs[grp]
        rows.append({
            "SourcePrecinct":        df["Precinct"].iloc[i],
            "Muni":                  df["Muni"].iloc[i],
            "GroupID":               grp,
            "GroupTotalEDay":        int(gt),
            "GroupHarrisEDay":       int(gh),
            "GroupHarrisPct":        round(gh / gt, 6) if gt > 0 else np.nan,
            "GroupRegisteredVoters": int(gr),
            "GroupTurnoutPct":       round(gt / gr, 6) if gr > 0 else np.nan,
            "SourceTotalEDay":       int(arr_total[i]),
            "SourceHarrisEDay":      int(arr_harris[i]),
            "SourceRegisteredVoters":int(arr_reg[i]),
        })
    pd.DataFrame(rows).sort_values(
        ["GroupID", "Muni", "SourcePrecinct"]
    ).to_csv(os.path.join(OUTPUT_DIR, filename), index=False)

print("Writing crosswalk CSVs…")
make_crosswalk_label(best_pos["label"], "crosswalk_sweep_best_positive.csv")
make_crosswalk_label(best_neg["label"], "crosswalk_sweep_best_negative.csv")
make_crosswalk_label(sa_pos["label"],   "crosswalk_sa_best_positive.csv")
make_crosswalk_label(sa_neg["label"],   "crosswalk_sa_best_negative.csv")
print("Crosswalk CSVs saved.")


# ── Summary CSV ───────────────────────────────────────────────────────────────
summary_df = pd.DataFrame(results_summary)[
    ["title", "n_groups", "r2", "signed_r2", "slope"]]
summary_df["r2"]        = summary_df["r2"].round(4)
summary_df["signed_r2"] = summary_df["signed_r2"].round(4)
summary_df["slope"]     = summary_df["slope"].round(8)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "scenario_summary.csv"), index=False)

pd.DataFrame({"signed_r2": all_sr2}).to_csv(
    os.path.join(OUTPUT_DIR, "sweep_all_signed_r2.csv"), index=False)

print("\n── Scenario Summary ─────────────────────────────────────────────────")
print(summary_df.to_string(index=False))
print(f"\nAll files written to: {OUTPUT_DIR}")
print("Done.")
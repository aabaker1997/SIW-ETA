"""
Allegheny County, PA 2024 Presidential Race — Randomized Adjacency Control
============================================================================
Identical to the cross-municipal annealer EXCEPT the geographic adjacency
graph is replaced with a random regular graph of the same average degree.

Purpose: demonstrate that the wide correlation range achievable under real
geographic adjacency (+0.88 to -0.99) requires genuine geographic clustering
of large/small and red/blue precincts. If vote-total patterns were randomly
distributed across the county (as would be expected under random vote
injection), a randomized adjacency graph should produce a much narrower
achievable range.

The "contiguous" constraint under random adjacency is meaningless geographically
— it just means groups must be connected in the fake graph. The point is the
statistical control: same vote totals, same annealing algorithm, different
topology.

Parallel execution: pass WORKER_ID as a command-line argument (0, 1, 2, …).
Each worker writes to its own subdirectory and uses a different RNG seed.
Results can be compared across workers or aggregated manually.

Usage:
    python allegheny_randadj.py 0
    python allegheny_randadj.py 1
    python allegheny_randadj.py 2
    (launch in separate terminals)
"""

import sys
import geopandas as gpd
import networkx as nx
from libpysal.weights import Rook, Queen
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

# ── Worker ID ─────────────────────────────────────────────────────────────────
WORKER_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
RNG_SEED  = 42 + 1_000_003 * WORKER_ID

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
INPUT_FILE = os.path.join(DATA_DIR, "allegheny_results.csv")
SHP_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "shapefile")
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "randadj")
SHP_JOIN_COL    = "Muni_War_1"

OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"worker_{WORKER_ID:02d}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RNG = np.random.default_rng(RNG_SEED)

N_RANDOM_SAMPLES   = 500_000
MIN_GROUPS         = 400
# MAX_GROUPS set dynamically to len(df) after load

# ── Random adjacency config ───────────────────────────────────────────────────
# Target average degree for the randomized graph.
# Real Rook graph averages ~4-5 neighbours per precinct.
# We match that so the search space is comparable.
RAND_ADJ_DEGREE = 5   # average neighbours per precinct in random graph

# ── Annealing config ──────────────────────────────────────────────────────────
SA_STEPS            = 25_000_000
SA_REPORT_INTERVAL  =  1_000_000
SA_STAGNATION_LIMIT =  2_000_000

SEED_MODE      = "full_split"
N_CONTIG_SWEEP = 2_000
PICKLE_POS     = os.path.join(OUTPUT_DIR, "best_pos_label.npy")
PICKLE_NEG     = os.path.join(OUTPUT_DIR, "best_neg_label.npy")

T_END              = 1e-6
SA_STAGES = [
    (0.40, 1.0),
    (0.30, 0.1),
    (0.20, 0.01),
    (0.10, 1e-5),
]
N_CALIBRATION      = 2_000
TARGET_ACCEPT_RATE = 0.80

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
print(f"Worker {WORKER_ID} (seed {RNG_SEED})")
print("Loading CSV…")
df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip()
df = df[df["TotalEDay"] > 0].copy().reset_index(drop=True)
df["Muni"]      = df["Precinct"].apply(assign_muni)
df["_join_key"] = df["Precinct"].str.strip()
N = len(df)
MAX_GROUPS    = N
csv_lower_map = {k.lower(): k for k in df["_join_key"]}
print(f"  {N} precincts  (MIN_GROUPS={MIN_GROUPS}, MAX_GROUPS={MAX_GROUPS})")

# ── Load shapefile to get real average degree ─────────────────────────────────
# We load the real graph only to measure its average degree, then discard it.
print("Loading shapefile (to measure real graph degree)…")
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

try:
    w = Rook.from_dataframe(gdf, silence_warnings=True)
except Exception:
    w = Queen.from_dataframe(gdf, silence_warnings=True)

real_degrees = [len(v) for v in w.neighbors.values()]
real_avg_degree = float(np.mean(real_degrees))
print(f"  Real Rook graph avg degree: {real_avg_degree:.2f}")
print(f"  Random graph target degree: {RAND_ADJ_DEGREE}")
print()

# ── Build RANDOMIZED adjacency graph ─────────────────────────────────────────
print("Building randomized adjacency graph…")
_adj_mutable = {i: set() for i in range(N)}

# Assign each precinct RAND_ADJ_DEGREE random neighbours (symmetric)
for i in range(N):
    candidates = [j for j in range(N) if j != i and i not in _adj_mutable[j]]
    n_needed   = max(0, RAND_ADJ_DEGREE - len(_adj_mutable[i]))
    if n_needed > 0 and candidates:
        chosen = RNG.choice(
            candidates,
            size=min(n_needed, len(candidates)),
            replace=False
        )
        for j in chosen:
            _adj_mutable[i].add(j)
            _adj_mutable[j].add(i)

_adj = {i: frozenset(v) for i, v in _adj_mutable.items()}
actual_avg = float(np.mean([len(v) for v in _adj.values()]))
print(f"  Randomized graph avg degree: {actual_avg:.2f}")

# Verify the random graph is connected (important for full_split seed to work)
visited = set(); stack = [0]
while stack:
    u = stack.pop()
    if u in visited: continue
    visited.add(u)
    stack.extend(_adj[u] - visited)
n_components = N - len(visited)
if n_components > 0:
    print(f"  WARNING: {n_components} isolated nodes — adding fallback edges")
    isolated = [i for i in range(N) if i not in visited]
    for i in isolated:
        j = int(RNG.integers(N))
        while j == i: j = int(RNG.integers(N))
        _adj_mutable[i].add(j); _adj_mutable[j].add(i)
        visited.add(i)
    _adj = {i: frozenset(v) for i, v in _adj_mutable.items()}
print(f"  Graph connected: {len(visited) == N}")
print()

# ── Vote arrays ───────────────────────────────────────────────────────────────
arr_total  = df["TotalEDay"].values.astype(np.float64)
arr_harris = df["HarrisEDay"].values.astype(np.float64)


# ── BFS and removability (always full BFS — no AP shortcut) ──────────────────
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
    return _bfs_connected(remaining, next(iter(nb_in)))


# ── Flat label state ──────────────────────────────────────────────────────────
def _init_label_full_split() -> np.ndarray:
    return np.arange(N, dtype=np.int32)


def _compact_labels(label: np.ndarray) -> np.ndarray:
    unique = np.unique(label)
    remap  = np.empty(label.max() + 1, dtype=np.int32)
    for new_id, old_id in enumerate(unique):
        remap[old_id] = new_id
    return remap[label]


def _repair_contiguity(label: np.ndarray) -> np.ndarray:
    new_label = label.copy()
    next_id   = int(label.max()) + 1
    for grp in np.unique(label):
        members    = list(np.where(label == grp)[0])
        if len(members) <= 1: continue
        member_set = set(members); visited = set()
        first = True
        for start in members:
            if start in visited: continue
            component = {start}; queue = [start]; visited.add(start)
            while queue:
                u = queue.pop()
                for v in _adj[u]:
                    if v in member_set and v not in visited:
                        visited.add(v); component.add(v); queue.append(v)
            if first:
                first = False
            else:
                for node in component:
                    new_label[node] = next_id
                next_id += 1
    return _compact_labels(new_label)


def _build_boundary(label: np.ndarray) -> set:
    boundary = set()
    for i in range(N):
        li = label[i]
        for j in _adj[i]:
            if label[j] != li:
                boundary.add(i)
                break
    return boundary


def _update_boundary(label, boundary, node, old_group, new_group):
    for i in {node} | _adj[node]:
        li = label[i]
        if any(label[j] != li for j in _adj[i]):
            boundary.add(i)
        else:
            boundary.discard(i)


def _fast_sr2_label(label: np.ndarray) -> float:
    n_grps  = int(label.max()) + 1
    totals  = np.bincount(label, weights=arr_total,  minlength=n_grps)
    harris  = np.bincount(label, weights=arr_harris, minlength=n_grps)
    mask    = totals > 0
    x       = totals[mask]; y = harris[mask] / x
    n = float(len(x))
    if n < 2: return 0.0
    xb = x.mean(); yb = y.mean()
    cov  = (x * y).mean() - xb * yb
    varx = (x * x).mean() - xb * xb
    vary = (y * y).mean() - yb * yb
    d = varx * vary
    if d <= 0: return 0.0
    return float(np.sign(cov) * cov * cov / d)


def _random_label_fast() -> np.ndarray:
    k = int(RNG.integers(MIN_GROUPS, MAX_GROUPS + 1))
    return RNG.integers(0, k, size=N, dtype=np.int32)


def _random_label() -> np.ndarray:
    """Contiguous BFS partition on the RANDOM adjacency graph."""
    k     = int(RNG.integers(MIN_GROUPS, MAX_GROUPS + 1))
    perm  = RNG.permutation(N)
    seeds = perm[:k]
    label = np.full(N, -1, dtype=np.int32)
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
                        label[nb] = gid; unassigned.discard(nb)
                        frontier[gid].append(nb); progress = True; break
                if progress: break
            if progress: break
        if not progress:
            for node in list(unassigned):
                for nb in _adj[node]:
                    if label[nb] != -1:
                        label[node] = label[nb]; unassigned.discard(node); break
            if unassigned:
                for node in unassigned: label[node] = 0
                unassigned.clear()
    return label


def _perturb_label(label, boundary, n_groups):
    if not boundary:
        return label, boundary, n_groups
    bd_list = list(boundary)
    node    = int(bd_list[int(RNG.integers(len(bd_list)))])
    src_grp = int(label[node])
    src_members = set(np.where(label == src_grp)[0])
    adj_groups  = {int(label[nb]) for nb in _adj[node] if label[nb] != src_grp}
    can_merge = (bool(adj_groups) and
                 (len(src_members) > 1 or n_groups > MIN_GROUPS))
    can_split = (n_groups < MAX_GROUPS and len(src_members) > 1)
    ops = []
    if can_merge: ops.append("merge")
    if can_split: ops.append("split")
    if not ops: return label, boundary, n_groups
    op = ops[int(RNG.integers(len(ops)))]
    if op == "merge":
        if len(src_members) > 1 and not _removable(src_members, node):
            return label, boundary, n_groups
        dst_grp   = int(RNG.choice(list(adj_groups)))
        new_label = label.copy()
        new_label[node] = dst_grp
        new_n = n_groups - (1 if len(src_members) == 1 else 0)
        if new_n < MIN_GROUPS: return label, boundary, n_groups
        new_boundary = set(boundary)
        _update_boundary(new_label, new_boundary, node, src_grp, dst_grp)
        return new_label, new_boundary, new_n
    else:
        if not _removable(src_members, node): return label, boundary, n_groups
        new_grp   = int(label.max()) + 1
        new_label = label.copy()
        new_label[node] = new_grp
        new_boundary = set(boundary)
        _update_boundary(new_label, new_boundary, node, src_grp, new_grp)
        return new_label, new_boundary, n_groups + 1


# ── T_START calibration ───────────────────────────────────────────────────────
def _calibrate_T_start(seed_label, seed_bd, seed_n, maximize, label_str):
    deltas = []
    lbl = seed_label.copy(); bd = set(seed_bd); n = seed_n
    for _ in range(N_CALIBRATION):
        new_lbl, new_bd, new_n = _perturb_label(lbl, bd, n)
        if new_lbl is not lbl:
            delta = _fast_sr2_label(new_lbl) - _fast_sr2_label(lbl)
            sd    = delta if maximize else -delta
            if sd < 0: deltas.append(abs(sd))
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


# ── Annealer ──────────────────────────────────────────────────────────────────
def anneal_label(seed_label, target, label_str):
    maximize      = (target == "pos")
    current_label = _repair_contiguity(_compact_labels(seed_label.copy()))
    n_after_repair = int(np.unique(current_label).size)
    current_sr2   = _fast_sr2_label(current_label)
    current_n     = n_after_repair
    current_bd    = _build_boundary(current_label)
    best_label    = current_label.copy()
    best_sr2      = current_sr2
    t_start       = time.time()
    step_total    = 0
    steps_since_improve = 0

    print(f"  Starting {label_str} anneal  sr²={current_sr2:+.4f}  "
          f"n_groups={current_n}  (seed had {int(np.unique(seed_label).size)}, "
          f"repaired to {n_after_repair})")

    T_START_cal = _calibrate_T_start(
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
        n_accept  = 0; n_improve = 0; stagnated = False

        print(f"  [{label_str}] Stage {stage_idx+1}: "
              f"{stage_steps:,} steps  T_start={stage_T:.5f}")

        for _ in range(stage_steps):
            step_total += 1; steps_since_improve += 1
            new_label, new_bd, new_n = _perturb_label(
                current_label, current_bd, current_n)
            if new_label is current_label:
                T *= decay; continue
            new_sr2      = _fast_sr2_label(new_label)
            delta        = new_sr2 - current_sr2
            signed_delta = delta if maximize else -delta
            if signed_delta >= 0:
                current_label = new_label; current_sr2 = new_sr2
                current_bd = new_bd; current_n = new_n
                n_accept += 1; n_improve += 1
                if (maximize and new_sr2 > best_sr2) or \
                   (not maximize and new_sr2 < best_sr2):
                    best_label = new_label.copy(); best_sr2 = new_sr2
                    steps_since_improve = 0
            else:
                if math.log(RNG.random()) < signed_delta / T:
                    current_label = new_label; current_sr2 = new_sr2
                    current_bd = new_bd; current_n = new_n
                    n_accept += 1
            T *= decay
            if step_total % SA_REPORT_INTERVAL == 0:
                elapsed   = time.time() - t_start
                rate      = step_total / elapsed
                remaining = (SA_STEPS - step_total) / rate
                print(f"    step {step_total/1e6:5.1f}M  T={T:.2e}  "
                      f"accept={n_accept/SA_REPORT_INTERVAL:.2%}  "
                      f"improve={n_improve/SA_REPORT_INTERVAL:.2%}  "
                      f"stagnant={steps_since_improve/1e6:.2f}M  "
                      f"n_grps={current_n}  "
                      f"cur={current_sr2:+.4f}  best={best_sr2:+.4f}  "
                      f"eta={remaining/3600:.2f}h")
                n_accept = 0; n_improve = 0
            if steps_since_improve >= SA_STAGNATION_LIMIT:
                print(f"  [{label_str}] Stagnation at step {step_total:,} — "
                      f"best={best_sr2:+.4f}  Terminating early.")
                stagnated = True; break
        if stagnated: break

    elapsed = time.time() - t_start
    print(f"  {label_str} done in {elapsed/3600:.2f}h ({step_total:,} steps) "
          f"— best={best_sr2:+.4f}  n_groups={np.unique(best_label).size}")
    return {"signed_r2": best_sr2, "label": best_label}


# ── Baseline ──────────────────────────────────────────────────────────────────
print("Baseline (full granularity)…")
label_baseline = _init_label_full_split()
baseline_sr2   = _fast_sr2_label(label_baseline)
print(f"  signed r² = {baseline_sr2:+.3f}  (n={N})\n")


# ── Random sweep ──────────────────────────────────────────────────────────────
print(f"Random sweep: {N_RANDOM_SAMPLES:,} unconstrained samples…")
best_pos = {"signed_r2": -999, "label": None}
best_neg = {"signed_r2":  999, "label": None}
all_sr2  = []

with tqdm(total=N_RANDOM_SAMPLES, unit="sample", ncols=80) as pbar:
    for _ in range(N_RANDOM_SAMPLES):
        lbl = _random_label_fast()
        sr2 = _fast_sr2_label(lbl)
        all_sr2.append(sr2)
        if sr2 > best_pos["signed_r2"]:
            best_pos = {"signed_r2": sr2, "label": lbl.copy()}
        if sr2 < best_neg["signed_r2"]:
            best_neg = {"signed_r2": sr2, "label": lbl.copy()}
        pbar.update(1)

print(f"\nSweep best pos: {best_pos['signed_r2']:+.3f}")
print(f"Sweep best neg: {best_neg['signed_r2']:+.3f}")
print(f"Mean: {np.mean(all_sr2):+.3f}  Std: {np.std(all_sr2):.3f}\n")


# ── Seed selection ────────────────────────────────────────────────────────────
print(f"Seed mode: {SEED_MODE}")
if SEED_MODE == "full_split":
    seed_pos = _init_label_full_split()
    seed_neg = _init_label_full_split()
    print(f"  Seeds: full split ({N} groups each)")
elif SEED_MODE == "contiguous_sweep":
    print(f"  Running {N_CONTIG_SWEEP:,} contiguous samples…")
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
    seed_pos = cs_best_pos["label"]; seed_neg = cs_best_neg["label"]
elif SEED_MODE == "from_pickle":
    seed_pos = np.load(PICKLE_POS); seed_neg = np.load(PICKLE_NEG)
else:
    raise ValueError(f"Unknown SEED_MODE: {SEED_MODE!r}")
print()


# ── Annealing ─────────────────────────────────────────────────────────────────
print("Simulated annealing: positive target…")
sa_pos = anneal_label(seed_pos, target="pos", label_str="Positive")
np.save(PICKLE_POS, sa_pos["label"])
print(f"  Saved to {PICKLE_POS}\n")

print("Simulated annealing: negative target…")
sa_neg = anneal_label(seed_neg, target="neg", label_str="Negative")
np.save(PICKLE_NEG, sa_neg["label"])
print(f"  Saved to {PICKLE_NEG}\n")


# ── Summary ───────────────────────────────────────────────────────────────────
summary = pd.DataFrame([
    {"config": "baseline",      "signed_r2": baseline_sr2,          "n_groups": N},
    {"config": "sweep_pos",     "signed_r2": best_pos["signed_r2"],  "n_groups": None},
    {"config": "sweep_neg",     "signed_r2": best_neg["signed_r2"],  "n_groups": None},
    {"config": "sa_pos",        "signed_r2": sa_pos["signed_r2"],    "n_groups": int(np.unique(sa_pos["label"]).size)},
    {"config": "sa_neg",        "signed_r2": sa_neg["signed_r2"],    "n_groups": int(np.unique(sa_neg["label"]).size)},
])
summary["signed_r2"] = summary["signed_r2"].round(4)
summary["worker_id"] = WORKER_ID
summary["rand_adj_degree"] = RAND_ADJ_DEGREE

print("── Summary ──────────────────────────────────────────────────────────")
print(summary.to_string(index=False))

summary.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)
pd.DataFrame({"signed_r2": all_sr2}).to_csv(
    os.path.join(OUTPUT_DIR, "sweep_sr2.csv"), index=False)

print(f"\nAll files written to: {OUTPUT_DIR}")
print("Done.")
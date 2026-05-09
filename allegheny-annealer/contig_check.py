import geopandas as gpd
import pandas as pd
import networkx as nx
from libpysal.weights import Queen
import os


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
BASE_DIR = r"os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")"
GDF_PATH       = os.path.join(BASE_DIR, "contiguity_graph.gpkg")
ASSIGNMENT_CSV = os.path.join(BASE_DIR, "precinctboundaries.csv")

PRECINCT_COL = "Muni_War_1"   # use the real precinct name column in CSV and GDF
GROUP_COL    = "OBJECTID"     # unique assignment ID
MUNI_COL     = "Muni_War_1"   # for checking muni splits

# Optional: relax small articulation points
ALLOW_SINGLE_PREC_BRIDGES = True

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
gdf = gpd.read_file(GDF_PATH)
assign = pd.read_csv(ASSIGNMENT_CSV)

# normalize keys
assign[PRECINCT_COL] = assign[PRECINCT_COL].str.strip()
gdf[PRECINCT_COL] = gdf[PRECINCT_COL].str.strip()

merged = gdf.merge(assign, on=PRECINCT_COL, how="left", indicator=True)
missing = merged[merged["_merge"] != "both"]
if not missing.empty:
    raise RuntimeError(f"{len(missing)} precincts missing assignment")

print("✓ Assignment covers all precincts")

# ─────────────────────────────────────────────────────────────
# BUILD CONTIGUITY GRAPH
# ─────────────────────────────────────────────────────────────
w = Queen.from_dataframe(merged, silence_warnings=True)
G = nx.Graph()
G.add_nodes_from(merged.index)
for i, nbrs in w.neighbors.items():
    for j in nbrs:
        G.add_edge(i, j)

# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def is_group_reachable(idx_list):
    """
    Check if a set of precincts could be merged by the annealer
    given contiguity constraints.
    """
    sub = G.subgraph(idx_list)
    if nx.is_connected(sub):
        return True

    # optionally allow single-precinct bridges
    if ALLOW_SINGLE_PREC_BRIDGES:
        # remove articulation points with degree 1 and re-test connectivity
        sub_copy = sub.copy()
        for node in list(nx.articulation_points(sub)):
            if sub.degree[node] == 1:
                sub_copy.remove_node(node)
        return nx.is_connected(sub_copy)

    return False

def check_all_groups():
    failures = []
    for gid, rows in merged.groupby(GROUP_COL):
        idx_list = rows.index.tolist()
        if len(idx_list) < 2:
            continue
        if not is_group_reachable(idx_list):
            failures.append(gid)
    return failures

def check_muni_splits():
    """
    Confirm whether any municipality is forced across multiple groups
    (diagnostic only, not a hard failure unless your annealer forbids splits)
    """
    splits = (
        merged.groupby([MUNI_COL, GROUP_COL])
              .size()
              .reset_index()
              .groupby(MUNI_COL)
              .size()
    )
    return splits[splits > 1]

# ─────────────────────────────────────────────────────────────
# RUN CHECKS
# ─────────────────────────────────────────────────────────────
contig_fail = check_all_groups()
if contig_fail:
    print(f"❌ Some groups in the real assignment are NOT reachable: {contig_fail}")
else:
    print("✓ All groups in the real assignment are reachable under current contiguity rules")

muni_splits = check_muni_splits()
print(f"ℹ Municipalities split across groups (diagnostic only): {len(muni_splits)}")
print(muni_splits.head(10))
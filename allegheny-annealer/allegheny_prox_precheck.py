"""
Allegheny County Contiguity Pre-Check
======================================
Loads the precinct shapefile, joins to the election results CSV on
Muni_War_1 → Precinct, builds a per-municipality contiguity graph,
and reports:
  - Join mismatches (shapefile records with no CSV match and vice versa)
  - Precincts with no neighbours within their municipality (isolated nodes)
  - Municipalities whose precincts form more than one connected component
    (i.e. the municipality is non-contiguous like O'Hara)
  - For each disconnected municipality, lists which precincts are in which component
  - Precincts that only touch precincts from OTHER municipalities
    (true exclaves that can never be merged with anyone in their own muni)

Outputs:
  contiguity_precheck_report.txt  — full human-readable report
  contiguity_graph.gpkg           — shapefile with component IDs added
                                    (open in QGIS to visually verify)
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from libpysal.weights import Queen, Rook
import os
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
SHP_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "shapefile")
CSV_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")\allegheny_results.csv
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
SHP_JOIN_COL = "Muni_War_1"   # shapefile column to match against CSV "Precinct"

os.makedirs(OUTPUT_DIR, exist_ok=True)
report_lines = []

def log(s=""):
    print(s)
    report_lines.append(s)


# ── Load data ─────────────────────────────────────────────────────────────────
log("Loading shapefile…")
gdf = gpd.read_file(SHP_DIR)
log(f"  Shapefile rows:   {len(gdf)}")
log(f"  CRS:              {gdf.crs}")
log(f"  Columns:          {list(gdf.columns)}")
log()

log("Loading CSV…")
csv = pd.read_csv(CSV_PATH)
csv.columns = csv.columns.str.strip()
log(f"  CSV rows:         {len(csv)}")
log()

# ── Normalise join keys ───────────────────────────────────────────────────────
# The shapefile uses inconsistent capitalisation and occasional abbreviations
# vs. the CSV. Strategy:
#   1. Build a lowercase lookup from CSV keys so we can match case-insensitively.
#   2. Apply a small manual map for known abbreviation mismatches
#      (Springdal Br → Springdale Br, Ohara → O'Hara).
#   3. Any shapefile key that still doesn't match after normalisation is flagged.

csv["_join_key"] = csv["Precinct"].str.strip()
csv_lower_map = {k.lower(): k for k in csv["_join_key"]}  # lowercase → canonical CSV key

# Manual overrides for abbreviation/punctuation differences.
# Keys are lowercase prefixes to replace; values are their CSV equivalents.
MANUAL_MAP = {
    "springdal br": "springdale br",   # shapefile truncates "Springdale"
    "ohara":        "o'hara",          # shapefile drops the apostrophe
}

def normalise_shp_key(raw: str) -> str:
    """Map a raw shapefile Muni_War_1 value to its canonical CSV Precinct string."""
    s  = raw.strip()
    sl = s.lower()
    # Apply prefix-based manual fixes before CSV lookup
    for prefix, replacement in MANUAL_MAP.items():
        if sl.startswith(prefix):
            sl = replacement + sl[len(prefix):]
            break
    # Look up in CSV (case-insensitive)
    return csv_lower_map.get(sl, s)   # fall back to original if no match

gdf["_join_key"] = gdf[SHP_JOIN_COL].apply(normalise_shp_key)

shp_keys = set(gdf["_join_key"])
csv_keys  = set(csv["_join_key"])

only_shp = shp_keys - csv_keys
only_csv = csv_keys - shp_keys

log("═" * 70)
log("JOIN DIAGNOSTICS")
log("═" * 70)
log(f"Shapefile unique Muni_War_1 values: {len(set(gdf[SHP_JOIN_COL].str.strip()))}")
log(f"CSV unique Precinct values:         {len(csv_keys)}")
log(f"After normalisation — unmatched shapefile keys: {len(only_shp)}")
log()

if only_shp:
    log(f"Still in shapefile but NOT in CSV after normalisation ({len(only_shp)}):")
    for k in sorted(only_shp):
        # show original value for debugging
        orig = gdf.loc[gdf["_join_key"] == k, SHP_JOIN_COL].iloc[0]
        log(f"  '{k}'  (original: '{orig}')")
else:
    log("✓ All shapefile keys matched to CSV after normalisation.")
log()

if only_csv:
    log(f"In CSV but NOT in shapefile ({len(only_csv)}):")
    for k in sorted(only_csv):
        log(f"  '{k}'")
else:
    log("✓ All CSV keys found in shapefile.")
log()

# ── Merge ─────────────────────────────────────────────────────────────────────
# Left join: keep all shapefile rows, attach CSV data where available
merged = gdf.merge(csv[["_join_key", "TotalEDay", "HarrisEDay", "HarrisPct"]],
                   on="_join_key", how="left")
log(f"Merged GDF rows: {len(merged)}")
n_unmatched = merged["TotalEDay"].isna().sum()
if n_unmatched:
    log(f"  WARNING: {n_unmatched} shapefile rows have no CSV match — "
        f"they will be excluded from contiguity analysis.")
log()

# Drop unmatched for contiguity work
matched = merged[merged["TotalEDay"].notna()].copy().reset_index(drop=True)

# ── Assign municipality labels (same logic as main script) ────────────────────
# Re-use the MULTI_PRECINCT_MUNIS list from the main script
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

# Precincts that must remain singletons regardless of prefix matching —
# either genuinely separate municipalities whose names are substrings of
# larger ones, or true geographic exclaves with no intra-muni neighbours.
SINGLETON_OVERRIDES = {
    "Rosslyn Farms",   # borough entirely surrounded by Ross Twp; not part of Ross
}

def assign_muni(precinct: str) -> str:
    if precinct in SINGLETON_OVERRIDES:
        return precinct   # force singleton — won't appear in multi_munis
    for muni in MULTI_PRECINCT_MUNIS:
        if precinct.startswith(muni):
            return muni
    return precinct

matched["Muni"] = matched["_join_key"].apply(assign_muni)

# ── Build Queen contiguity weights (shared edge OR vertex = neighbours) ───────
log("Building Queen contiguity graph…")
# Queen: precincts sharing any boundary point are neighbours.
# Use Rook (shared edge only) as a stricter alternative — comment/uncomment as needed.
try:
    w = Queen.from_dataframe(matched, silence_warnings=True)
except Exception as e:
    log(f"  Queen failed ({e}), falling back to Rook…")
    w = Rook.from_dataframe(matched, silence_warnings=True)

log(f"  Islands (no neighbours at all): {len(w.islands)}")
if w.islands:
    for idx in w.islands:
        log(f"    Row {idx}: {matched.loc[idx, '_join_key']} (Muni: {matched.loc[idx, 'Muni']})")
log()

# ── Per-municipality contiguity analysis ──────────────────────────────────────
log("═" * 70)
log("PER-MUNICIPALITY CONTIGUITY ANALYSIS")
log("═" * 70)

muni_counts = matched.groupby("Muni").size()
multi_munis = muni_counts[muni_counts > 1].index.tolist()

component_records = []   # for GeoPackage output
problem_munis = []

for muni in sorted(multi_munis):
    muni_rows = matched[matched["Muni"] == muni]
    muni_idx  = muni_rows.index.tolist()

    if len(muni_idx) < 2:
        continue

    # Build intra-municipality subgraph
    G = nx.Graph()
    G.add_nodes_from(muni_idx)
    for i in muni_idx:
        for j in w.neighbors.get(i, []):
            if j in set(muni_idx):
                G.add_edge(i, j)

    components = list(nx.connected_components(G))
    n_comp     = len(components)

    # Assign component IDs back to rows
    for comp_id, comp in enumerate(components):
        for row_idx in comp:
            component_records.append({
                "row_idx":   row_idx,
                "Muni":      muni,
                "component": comp_id,
                "n_components": n_comp,
            })

    if n_comp > 1:
        problem_munis.append(muni)
        log(f"⚠  {muni}: {len(muni_idx)} precincts, {n_comp} disconnected components")
        for comp_id, comp in enumerate(sorted(components, key=len, reverse=True)):
            precinct_names = sorted(matched.loc[list(comp), "_join_key"].tolist())
            log(f"   Component {comp_id + 1} ({len(comp)} precincts):")
            for name in precinct_names:
                # check if this precinct has ANY intra-muni neighbours
                nb_in_muni = [j for j in w.neighbors.get(
                    matched[matched["_join_key"] == name].index[0], [])
                    if j in set(muni_idx)]
                flag = " ← NO INTRA-MUNI NEIGHBOURS (exclave)" if not nb_in_muni else ""
                log(f"     {name}{flag}")
        log()

log()
if problem_munis:
    log(f"Municipalities with disconnected components ({len(problem_munis)}):")
    for m in problem_munis:
        log(f"  {m}")
else:
    log("✓ All multi-precinct municipalities are spatially contiguous.")
log()

# ── True exclaves: precincts with NO intra-municipality neighbours ─────────────
log("═" * 70)
log("EXCLAVE CHECK (precincts with no intra-municipality neighbours)")
log("═" * 70)
exclaves = []
for muni in multi_munis:
    muni_set = set(matched[matched["Muni"] == muni].index.tolist())
    for idx in muni_set:
        nb_in_muni = [j for j in w.neighbors.get(idx, []) if j in muni_set]
        if not nb_in_muni:
            exclaves.append((muni, matched.loc[idx, "_join_key"], idx))

if exclaves:
    log(f"Found {len(exclaves)} exclave precinct(s):")
    for muni, name, idx in sorted(exclaves):
        # show what municipalities its neighbours belong to instead
        nb_munis = sorted(set(matched.loc[j, "Muni"]
                              for j in w.neighbors.get(idx, [])))
        log(f"  {name}  (Muni: {muni})  →  neighbours are in: {nb_munis}")
else:
    log("✓ No true exclaves found.")
log()

# ── Attach component info to GDF and save ─────────────────────────────────────
comp_df = pd.DataFrame(component_records).set_index("row_idx")
matched["intra_muni_component"] = comp_df["component"].reindex(matched.index).fillna(0).astype(int)
matched["n_components"]         = comp_df["n_components"].reindex(matched.index).fillna(1).astype(int)

out_gpkg = os.path.join(OUTPUT_DIR, "contiguity_graph.gpkg")
matched.drop(columns=["_join_key"]).to_file(out_gpkg, driver="GPKG")
log(f"GeoPackage written: {out_gpkg}")
log("  (open in QGIS, style by 'intra_muni_component' to visually verify)")
log()

# ── Write report ──────────────────────────────────────────────────────────────
report_path = os.path.join(OUTPUT_DIR, "contiguity_precheck_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
log(f"Report written: {report_path}")
log("Done.")
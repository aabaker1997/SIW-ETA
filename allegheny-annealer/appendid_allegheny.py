"""
Allegheny County — Crosswalk Join Script
=========================================
Joins the best SA crosswalk GroupIDs (muni-bounded and cross-municipal)
to the main results file, outputting a single CSV with one row per source
precinct and GroupID columns for each configuration.

You can then pivot/groupby in Excel, QGIS, or Python to calculate
HarrisPct, MailPct, AllVotesPct etc. for each grouping.

Output: allegheny_results_crosswalked.csv
"""

import pandas as pd
import os

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = r"os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")"
INPUT_FILE = os.path.join(DATA_DIR, "allegheny_results.csv")

# Crosswalk CSVs to join — add or remove as needed
CROSSWALKS = {
    "muni_sa_pos": r"os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")\crosswalk_sa_best_positive.csv",
    "muni_sa_neg": r"os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")\crosswalk_sa_best_negative.csv",
    "xmuni_sa_pos": r"os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "crossmuni")\crosswalk_sa_best_positive.csv",
    "xmuni_sa_neg": r"os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "crossmuni")\crosswalk_sa_best_negative.csv",
}

OUTPUT_FILE = os.path.join(DATA_DIR, "allegheny_results_crosswalked.csv")

# ── Load main results ─────────────────────────────────────────────────────────
print("Loading main results…")
df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip()
df = df[df["TotalEDay"] > 0].copy().reset_index(drop=True)
df["_join_key"] = df["Precinct"].str.strip()
print(f"  {len(df)} precincts")

# ── Join each crosswalk ───────────────────────────────────────────────────────
for label, path in CROSSWALKS.items():
    if not os.path.exists(path):
        print(f"  WARNING: {label} crosswalk not found at {path} — skipping")
        continue

    print(f"Joining {label}…")
    cw = pd.read_csv(path)
    cw.columns = cw.columns.str.strip()

    # Muni-bounded crosswalks use "SourcePrecinct", crossmuni uses same
    # Both have a group identifier — use GroupID if present, else AggregatedGroup
    if "GroupID" in cw.columns:
        grp_col = "GroupID"
    elif "AggregatedGroup" in cw.columns:
        grp_col = "AggregatedGroup"
    else:
        print(f"  WARNING: no group column found in {label} — skipping")
        continue

    cw = cw[["SourcePrecinct", grp_col]].copy()
    cw.columns = ["_join_key", f"GroupID_{label}"]
    cw["_join_key"] = cw["_join_key"].str.strip()

    before = len(df)
    df = df.merge(cw, on="_join_key", how="left")
    n_joined = df[f"GroupID_{label}"].notna().sum()
    print(f"  {n_joined}/{before} precincts joined")

# ── Clean up and output ───────────────────────────────────────────────────────
df = df.drop(columns=["_join_key"])

print(f"\nWriting {OUTPUT_FILE}…")
df.to_csv(OUTPUT_FILE, index=False)
print(f"Done — {len(df)} rows, {len(df.columns)} columns")
print(f"\nColumns: {list(df.columns)}")

# ── Quick group summary for each crosswalk ────────────────────────────────────
print("\n── Group summaries ──────────────────────────────────────────────────")
for label in CROSSWALKS:
    col = f"GroupID_{label}"
    if col not in df.columns:
        continue
    n_groups  = df[col].nunique()
    n_singles = (df.groupby(col)[col].transform("count") == 1).sum()
    print(f"  {label}: {n_groups} groups, {n_singles} singleton precincts")
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import math
import os
import matplotlib.pyplot as plt
import re

# ------------------ CONFIG ------------------
MAJOR_PARTIES = [
    "Liberal", "Conservative", "NDP", "Green", "Bloc", "People's"
]

PARTY_KEYWORDS = {
    "Liberal": "Liberal",
    "Conservative": "Conservative",
    "NDP": "N.D.P.",
    "Green": "Green",
    "Bloc": "Bloc",
    "People's": "People's"
}

BASE_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

OUTPUT_DIRS = {
    "PartyTurnout": os.path.join(BASE_OUTPUT_DIR, "PartyTurnout"),
    "PartyVotes": os.path.join(BASE_OUTPUT_DIR, "PartyVotes"),
    "LRTurnout": os.path.join(BASE_OUTPUT_DIR, "LRTurnout"),
    "LRVotes": os.path.join(BASE_OUTPUT_DIR, "LRVotes")
}

TOP_N = {
    "PartyTurnout": 100,
    "PartyVotes": 100,
    "LRTurnout": 25,
    "LRVotes": 25
}

for d in OUTPUT_DIRS.values():
    os.makedirs(d, exist_ok=True)

# ------------------ HELPERS ------------------
def party_votes_by_precinct(df_d, keyword):
    temp = df_d[df_d["party"].str.contains(keyword, case=False, na=False)]
    return temp.groupby("poll")["votes"].sum().to_dict()

def safe_corr(x, y):
    mask = (~x.isna()) & (~y.isna()) & (~np.isinf(x)) & (~np.isinf(y))
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan, np.nan
    if np.isclose(x.std(), 0) or np.isclose(y.std(), 0):
        return np.nan, np.nan

    try:
        r, p = pearsonr(x, y)
        return r, p
    except Exception:
        return np.nan, np.nan

def safe_name(s):
    return re.sub(r"[^\w\-_.]", "_", str(s))

def save_scatter(x, y, xlabel, ylabel, title, r, p, outpath):
    """
    Save a scatter plot with a linear trendline.
    Handles NaNs, infinities, and unsorted data safely.
    """
    plt.figure(figsize=(6, 5))
    
    # Convert to float and filter invalid values
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = (~x.isna()) & (~y.isna()) & (~np.isinf(x)) & (~np.isinf(y))
    x_clean = x[mask].astype(float)
    y_clean = y[mask].astype(float)
    
    if len(x_clean) == 0:
        print(f"Warning: No valid data to plot for {title}")
        return
    
    # Scatter points
    plt.scatter(x_clean, y_clean, alpha=0.6)
    
    # Add linear trendline if enough points
    if len(x_clean) >= 2:
        m, b = np.polyfit(x_clean, y_clean, 1)
        x_sorted = np.sort(x_clean)
        plt.plot(x_sorted, m * x_sorted + b, color='red', lw=2)
    
    # Labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    # Stats annotation
    plt.text(
        0.05, 0.95,
        f"$r^2$ = {r**2 if not np.isnan(r) else 'NaN':.3f}\np = {p if not np.isnan(p) else 'NaN':.3g}",
        transform=plt.gca().transAxes,
        verticalalignment="top"
    )
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# ------------------ LOAD DATA ------------------
df = pd.read_csv(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "canada_2025.csv"),
    low_memory=False
)

df = df[
    [
        "Electoral District Name_English/Nom de circonscription_Anglais",
        "Polling Division Number/Numéro de section de vote",
        "Polling Division Name/Nom de section de vote",
        "Political Affiliation Name_English/Appartenance politique_Anglais",
        "Candidate Vote Count/Votes du candidat",
        "Electors for poll/Électeurs du bureau"
    ]
].copy()

df.columns = ["district", "poll", "poll_name", "party", "votes", "electors"]
df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0)
df["electors"] = pd.to_numeric(df["electors"], errors="coerce").fillna(0)

# ------------------ STORAGE FOR CORRELATIONS ------------------
results_party_turnout = []
results_party_totalvotes = []
results_lr_turnout = []
results_lr_totalvotes = []

# ------------------ MAIN CALCULATION ------------------
for district, df_d in df.groupby("district"):
    precinct_df = df_d.groupby("poll").agg(
        total_votes_precinct=("votes", "sum"),
        electors=("electors", "first"),
        poll_name=("poll_name", "first")
    ).reset_index()
    precinct_df["turnout"] = precinct_df["total_votes_precinct"] / precinct_df["electors"].replace(0, np.nan)

    # ------------------ PARTY ------------------
    for party in MAJOR_PARTIES:
        pv = party_votes_by_precinct(df_d, PARTY_KEYWORDS[party])
        precinct_df["party_votes"] = precinct_df["poll"].map(pv).fillna(0)
        precinct_df["party_perf"] = precinct_df["party_votes"] / precinct_df["total_votes_precinct"].replace(0, np.nan)

        # Votes correlation with filtering
        votes_filter = (
            ~precinct_df["poll"].str.contains("S/R", na=False) &
            ~precinct_df["poll"].str.match(r"6") &
            ~precinct_df["poll_name"].str.contains("Mobile poll/Bureau itinérant", na=False)
        )
        r_total, p_total = safe_corr(
            precinct_df.loc[votes_filter, "party_perf"],
            precinct_df.loc[votes_filter, "total_votes_precinct"]
        )
        results_party_totalvotes.append({
            "district": district,
            "party": party,
            "r": r_total,
            "r2": r_total**2 if not math.isnan(r_total) else np.nan,
            "p": p_total
        })

        # Turnout correlation: filter 0 < turnout <= 1
        filtered = precinct_df[(precinct_df["turnout"] > 0) & (precinct_df["turnout"] <= 1)]
        r_turn, p_turn = safe_corr(filtered["party_perf"], filtered["turnout"])
        results_party_turnout.append({
            "district": district,
            "party": party,
            "r": r_turn,
            "r2": r_turn**2 if not math.isnan(r_turn) else np.nan,
            "p": p_turn
        })

    # ------------------ LEFT/RIGHT ------------------
    has_bloc = df_d["party"].str.contains("Bloc", case=False, na=False).any()
    if not has_bloc:
        left_keys = ["Liberal", "NDP", "Green"]
        right_keys = ["Conservative", "People's"]
        precinct_df["left_votes"] = sum(precinct_df["poll"].map(party_votes_by_precinct(df_d, k)).fillna(0) for k in left_keys)
        precinct_df["right_votes"] = sum(precinct_df["poll"].map(party_votes_by_precinct(df_d, k)).fillna(0) for k in right_keys)
        precinct_df["left_perf"] = precinct_df["left_votes"] / precinct_df["total_votes_precinct"].replace(0, np.nan)
        precinct_df["right_perf"] = precinct_df["right_votes"] / precinct_df["total_votes_precinct"].replace(0, np.nan)

        # Votes
        for side in ["left", "right"]:
            votes_filter = (
                ~precinct_df["poll"].str.contains("S/R", na=False) &
                ~precinct_df["poll"].str.match(r"6") &
                ~precinct_df["poll_name"].str.contains("Mobile poll/Bureau itinérant", na=False)
            )
            r_total, p_total = safe_corr(
                precinct_df.loc[votes_filter, f"{side}_perf"],
                precinct_df.loc[votes_filter, "total_votes_precinct"]
            )
            results_lr_totalvotes.append({
                "district": district,
                "side": side,
                "r": r_total,
                "r2": r_total**2 if not math.isnan(r_total) else np.nan,
                "p": p_total
            })

        # Turnout
        filtered = precinct_df[(precinct_df["turnout"] > 0) & (precinct_df["turnout"] <= 1)]
        for side in ["left", "right"]:
            r_turn, p_turn = safe_corr(filtered[f"{side}_perf"], filtered["turnout"])
            results_lr_turnout.append({
                "district": district,
                "side": side,
                "r": r_turn,
                "r2": r_turn**2 if not math.isnan(r_turn) else np.nan,
                "p": p_turn
            })

# ------------------ TOP N FILTERING ------------------
def top_n(df_list, n):
    df = pd.DataFrame(df_list)
    return df.sort_values("r2", ascending=False).head(n)

res_turnout_df = top_n(results_party_turnout, TOP_N["PartyTurnout"])
res_total_df = top_n(results_party_totalvotes, TOP_N["PartyVotes"])
res_lr_turn_df = top_n(results_lr_turnout, TOP_N["LRTurnout"])
res_lr_total_df = top_n(results_lr_totalvotes, TOP_N["LRVotes"])

# ------------------ PLOT FUNCTION ------------------
def plot_top(df_top, x_col, y_col, xlabel, ylabel, category, is_lr=False):
    for _, row in df_top.iterrows():
        district = safe_name(row["district"])
        name = safe_name(row["party"]) if not is_lr else row["side"]
        df_district = df[df["district"] == row["district"]].copy()
        precinct_df = df_district.groupby("poll").agg(
            total_votes_precinct=("votes", "sum"),
            electors=("electors", "first"),
            poll_name=("poll_name", "first")
        ).reset_index()
        precinct_df["turnout"] = precinct_df["total_votes_precinct"] / precinct_df["electors"].replace(0, np.nan)

        if is_lr:
            left_keys = ["Liberal", "NDP", "Green"]
            right_keys = ["Conservative", "People's"]
            precinct_df["left_votes"] = sum(precinct_df["poll"].map(party_votes_by_precinct(df_district, k)).fillna(0) for k in left_keys)
            precinct_df["right_votes"] = sum(precinct_df["poll"].map(party_votes_by_precinct(df_district, k)).fillna(0) for k in right_keys)
            precinct_df["left_perf"] = precinct_df["left_votes"] / precinct_df["total_votes_precinct"].replace(0, np.nan)
            precinct_df["right_perf"] = precinct_df["right_votes"] / precinct_df["total_votes_precinct"].replace(0, np.nan)
            y = precinct_df[f"{row['side']}_perf"]
        else:
            pv = party_votes_by_precinct(df_district, row["party"])
            precinct_df["party_votes"] = precinct_df["poll"].map(pv).fillna(0)
            precinct_df["party_perf"] = precinct_df["party_votes"] / precinct_df["total_votes_precinct"].replace(0, np.nan)
            y = precinct_df["party_perf"]

        x = precinct_df[x_col]
        if "Votes" in category:
            votes_filter = (
                ~precinct_df["poll"].str.contains("S/R", na=False) &
                ~precinct_df["poll"].str.match(r"6") &
                ~precinct_df["poll_name"].str.contains("Mobile poll/Bureau itinérant", na=False)
            )
            x = x[votes_filter]
            y = y[votes_filter]
        else:  # Turnout
            mask = (precinct_df["turnout"] > 0) & (precinct_df["turnout"] <= 1)
            x = x[mask]
            y = y[mask]

        if x.empty or y.empty:
            continue

        save_scatter(
            x, y, xlabel, ylabel,
            f"{row['district']} – {name}",
            row["r"], row["p"],
            os.path.join(OUTPUT_DIRS[category], f"{district}_{name}.png")
        )

# ------------------ SAVE TOP PLOTS ------------------
plot_top(res_total_df, "total_votes_precinct", "party_perf", "Total Votes (Precinct)", "Vote Share", "PartyVotes")
plot_top(res_turnout_df, "turnout", "party_perf", "Turnout", "Vote Share", "PartyTurnout")
plot_top(res_lr_total_df, "total_votes_precinct", "left_perf", "Total Votes (Precinct)", "Vote Share", "LRVotes", is_lr=True)
plot_top(res_lr_turn_df, "turnout", "left_perf", "Turnout", "Vote Share", "LRTurnout", is_lr=True)

print("Done. Top plots saved to:")
for k, v in OUTPUT_DIRS.items():
    print(f"{k}: {v}")

# ------------------ SAVE TOP CORRELATIONS TO CSV ------------------
def save_top_correlations(df_top, category, is_lr=False):
    """
    Save top correlations to CSV.
    df_top: DataFrame with columns including district, party/side, r, r2, p
    category: string, one of ["PartyTurnout", "PartyVotes", "LRTurnout", "LRVotes"]
    is_lr: bool, whether this is left/right subset
    """
    df_out = df_top.copy()
    if is_lr:
        df_out.rename(columns={"side": "Party/Side", "district": "District"}, inplace=True)
    else:
        df_out.rename(columns={"party": "Party", "district": "District"}, inplace=True)
    
    # Keep only relevant columns
    cols = ["District", "Party" if not is_lr else "Party/Side", "r", "r2", "p"]
    df_out = df_out[cols]

    # Ensure output directory exists
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # Save CSV
    csv_path = os.path.join(BASE_OUTPUT_DIR, f"TopCorrelations_{category}.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"Top correlations for {category} saved to {csv_path}")

# ------------------ SAVE CSVS ------------------
save_top_correlations(res_turnout_df, "PartyTurnout", is_lr=False)
save_top_correlations(res_total_df, "PartyVotes", is_lr=False)
save_top_correlations(res_lr_turn_df, "LRTurnout", is_lr=True)
save_top_correlations(res_lr_total_df, "LRVotes", is_lr=True)


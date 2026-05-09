import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
import re
from collections import defaultdict
from fuzzywuzzy import fuzz

# Plotting function
def make_plot(df, x_var, y_var, plot_type):
    plt.figure(figsize=(10, 6))

    if plot_type == "kde":
        sns.kdeplot(data=df, x=x_var, y=y_var, cmap="mako", fill=True, thresh=0.05, levels='auto')
    elif plot_type == "hist2d":
        plt.hist2d(df[x_var], df[y_var], bins=100, cmap="viridis")
        plt.colorbar(label='Number of Precincts')
    elif plot_type == "hex":
        plt.hexbin(df[x_var], df[y_var], gridsize=50, cmap="plasma")
        plt.colorbar(label='Count')
    elif plot_type == "boxplot":
        sns.boxplot(data=df, x=x_var, y=y_var)
    else:
        print("Invalid plot type.")
        return

    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.title(f"{y_var} vs {x_var} ({plot_type})")
    plt.tight_layout()
    plt.show()

# Main execution
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="CSV file of election data")
    parser.add_argument("--office", help="Office to filter for (e.g. President, US Senate, etc.)")
    parser.add_argument("--district", help="Optional: filter for a specific district")
    parser.add_argument("--counties", help="Comma-separated list of counties to include")
    parser.add_argument("--plot", default="hex", choices=["kde", "hist2d", "hex", "boxplot"], help="Plot type")
    parser.add_argument("--minvotes", type=int, default=10, help="Minimum votes in a precinct to include")
    args = parser.parse_args()

    # Step 1: Read the CSV
    df = pd.read_csv(args.file)

    # Step 2: Normalize column names
    df.columns = [col.lower().strip() for col in df.columns]

    # --- Drop non-candidate entries ---
    non_candidate_pattern = r'(?:over|under|blank|total)'  # Use non-capturing group (?:)
    df = df[~df['candidate'].str.contains(non_candidate_pattern, na=False, regex=True, flags=re.IGNORECASE)]

    # Step 3: Filter to desired office
    offices = df['office'].dropna().unique()
    print("Available offices:", sorted(offices))

    if args.office:
        selected_office = args.office.strip()
    else:
        selected_office = input("Enter the office you want to analyze (e.g., 'President'): ").strip()

    df = df[df['office'].str.strip().str.lower() == selected_office.lower()]

    # --- Dynamically check for districts based on selected office ---
    if 'district' in df.columns:
        office_districts = df[df['office'].str.strip().str.lower() == selected_office.lower()]['district'].dropna().unique()

        if len(office_districts) > 0:
            office_districts = office_districts.astype(int).astype(str)
            print("\nAvailable districts:")
            for i, district in enumerate(office_districts):
                print(f"{i + 1}. {district}")

            while True:
                district_choice = input("Enter the district you want to filter by (e.g., 18): ").strip()
                try:
                    district_choice_float = float(district_choice)
                    if str(int(district_choice_float)) in office_districts:
                        df = df[df['district'] == district_choice_float]
                        print(f"Filtering data to district: {district_choice_float}")
                        break
                    else:
                        print("Invalid district number. Please try again.")
                except ValueError:
                    print("Invalid district number. Please enter a valid number.")

    # --- Remove hyphens from candidate and party fields ---
    df['candidate'] = df['candidate'].str.replace('-', '', regex=False)
    df['party'] = df['party'].str.replace('-', '', regex=False)

    # --- Step 4: Standardize party names (e.g., 'D' -> 'DEM') ---
    def normalize_party(party):
        party = str(party).strip().upper()
        if party in ['GOP', 'R', 'REP', 'REPUBLICAN']:
            return 'REP'
        elif party in ['D', 'DEM', 'DEMOCRATIC', 'DEMOCRAT']:
            return 'DEM'
        elif party in ['L', 'LIBERTARIAN', 'LIB', 'LBT']:
            return 'LBT'
        elif party in ['G', 'GREEN', 'GEN', 'GRN', 'PGP', 'PAC', 'PACIFIC GREEN']:
            return 'GRN'
        return party
    
    df['party'] = df['party'].apply(normalize_party)

    # Step 5: Strip whitespace and fill NaNs
    df["candidate"] = df["candidate"].fillna("").str.strip()
    df["party"] = df["party"].fillna("OTH").str.strip().str.upper()

    # 1. Flag rows that need manual review (NaN candidate and party name as candidate)
    df['needs_manual_check'] = df.apply(lambda row: pd.isna(row['candidate']) and isinstance(row['party'], str) and row['party'].strip() != "", axis=1)

    # 2. Swap candidate and party if party field contains valid party name and candidate is NaN
    def swap_candidate_party(row):
        recognizable_parties = {'DEM', 'REP', 'LBT', 'GRN', 'GOP', 'D', 'R', 'L', 
                        'DEMOCRATIC', 'REPUBLICAN', 'LIBERTARIAN', 'GREEN'}
        if pd.isna(row['candidate']) and row['party'] in recognizable_parties:
            row['candidate'] = row['party']
            row['party'] = 'OTH'
        return row

    df = df.apply(swap_candidate_party, axis=1)

    # 3. Normalize non-major parties
    def normalize_non_major_parties(party):
        non_major_parties = ['WRITEIN', 'WRITEINS', 'W', 'CONSTITUTION','NAN','NON','NP','WRI','ONA','PROGRESSIVE','WTP','PROGESSIVE','WE THE PEOPLE']
        if isinstance(party, str):
            party = party.strip().upper()
            if party in non_major_parties:
                return 'OTH'  # Standardize non-major parties as 'OTH'
        return party

    df['party'] = df['party'].apply(normalize_non_major_parties)

    valid_parties = {'DEM', 'REP', 'LBT', 'GRN', 'OTH'}


# Detect rows where candidate is blank or null and party looks like a candidate
    suspect_blank_candidate = df[
        (df['candidate'].fillna("").str.strip() == "") &
        (~df['party'].isna()) &
        (~df['party'].str.upper().isin(valid_parties))
]

# Combine with earlier 'flipped' logic
    suspect_flipped = df[
        (df['candidate'].str.upper().isin(valid_parties)) &
        (~df['party'].str.upper().isin(valid_parties)) &
        (~df['party'].isna()) &
        (~df['candidate'].isna())
]

# Merge both types
    suspect_rows = pd.concat([suspect_flipped, suspect_blank_candidate]).drop_duplicates(subset=['candidate', 'party'])

# Prompt user for correction
    manual_corrections = {}
    for _, row in suspect_rows.iterrows():
        cand_val = row['candidate'] if pd.notna(row['candidate']) else '[blank]'
        party_val = row['party']
        print(f"\nFor records where candidate = '{cand_val}' and party = '{party_val}', what should they be?")
        corrected_candidate = input("  → Enter corrected candidate name: ").strip()
        corrected_party = input("  → Enter corrected party (e.g., DEM, REP, OTH): ").strip().upper()
        manual_corrections[(row['candidate'], row['party'])] = (corrected_candidate, corrected_party)

# Apply the manual corrections
    def correct_misplaced(row):
        key = (row['candidate'], row['party'])
        if key in manual_corrections:
            row['candidate'], row['party'] = manual_corrections[key]
        return row

    df = df.apply(correct_misplaced, axis=1)

# Record count before drop
    before_drop = len(df)

# Drop rows where candidate is blank/null and party is 'OTH'
    df = df[~((df['candidate'].fillna('').str.strip() == '') & (df['party'] == 'OTH'))]

# Record count after drop
    after_drop = len(df)

# Print how many rows were dropped
    print(f"Dropped {before_drop - after_drop} row(s) where candidate was blank and party was 'OTH'.")

    # --- Candidate name consolidation by party ---
    print("\nConsolidating candidate names by party...")

    # Optional: print what mappings were created
    unique_candidates = df[['candidate', 'party']].drop_duplicates()
    print("\nAuto-normalized candidate names:")
    print(unique_candidates.to_string(index=False))

    # Pivot to wide format
    df_wide = df.pivot_table(index=['county', 'precinct'],
                             columns='party',
                             values='votes',
                             aggfunc='sum',
                             fill_value=0).reset_index()

    party_cols = [col for col in df_wide.columns if col not in ['county', 'precinct']]

    # Convert all candidate columns to numeric
    df_wide[party_cols] = df_wide[party_cols].apply(pd.to_numeric, errors='coerce')

    # Sum total votes per row
    df_wide['Total'] = df_wide[party_cols].sum(axis=1)

    # Drop rows where Total couldn't be calculated
    df_wide = df_wide.dropna(subset=['Total'])
    df_wide = df_wide[df_wide['Total'] >= args.minvotes]

    # Efficiently calculate all percentage columns at once
    pct_df = df_wide[party_cols].div(df_wide['Total'], axis=0).multiply(100)
    pct_df.columns = [f"{col}_pct" for col in pct_df.columns]

    # Concatenate the percentage columns all at once
    df_wide = pd.concat([df_wide, pct_df], axis=1)

    # Optional: de-fragment
    df_wide = df_wide.copy()
    
    # Try to find a 'registered voters' column from the original dataframe
    reg_col_candidates = ['registered voters', 'registration', 'registered']
    reg_voters_col = None
    for col in df.columns:
        for target in reg_col_candidates:
            if target.lower() in col.lower():
                reg_voters_col = col
                break
        if reg_voters_col:
            break

    if reg_voters_col:
        print(f"Found registration column: {reg_voters_col}")
        # Merge it into the wide dataframe on county + precinct
        reg_df = df[['county', 'precinct', reg_voters_col]].drop_duplicates()
        reg_df[reg_voters_col] = pd.to_numeric(reg_df[reg_voters_col], errors='coerce')
        df_wide = pd.merge(df_wide, reg_df, on=['county', 'precinct'], how='left')

        df_wide['turnout'] = df_wide['Total'] / df_wide[reg_voters_col]
    else:
        print("No registration column found. Skipping turnout calculation.")
    
    # -----------------------
    # Diagnostic checks
    # -----------------------

    if 'turnout' in df_wide.columns:
        print("\n--- Turnout Diagnostic ---")
        # Check for precincts with turnout > 100%
        suspicious_turnout = df_wide[df_wide['turnout'] > 1.05]
        print(f"Precincts with >105% turnout: {len(suspicious_turnout)}")
        if not suspicious_turnout.empty:
            print(suspicious_turnout[['county', 'precinct', 'Total', reg_voters_col, 'turnout']].sort_values(by='turnout', ascending=False).head(10))

        # Summarize turnout distribution
        print("\nTurnout distribution stats:")
        print(df_wide['turnout'].describe(percentiles=[.5, .75, .9, .95, .99]))

# Ask user if they want to filter out records with more than a certain number of votes
    filter_out_votes = input("Do you want to filter out precincts with more than X votes? (yes/no): ").strip().lower()

    if filter_out_votes == 'yes':
        try:
        # Get the vote threshold from the user
            vote_threshold = int(input("Enter the vote threshold (e.g., 5000): "))
            df_wide = df_wide[df_wide['Total'] <= vote_threshold]  # Filter out rows with votes greater than threshold
            print(f"Records with more than {vote_threshold} votes have been removed.")
        except ValueError:
            print("Invalid input. Please enter a valid number for the vote threshold.")

# Check for high total vote outliers (after filtering if requested)
    print("\n--- Vote Total Diagnostic ---")
    extreme_votes = df_wide[df_wide['Total'] > 5000]
    print(f"Precincts with >5,000 votes: {len(extreme_votes)}")
    if not extreme_votes.empty:
        print(extreme_votes[['county', 'precinct', 'Total']].sort_values(by='Total', ascending=False).head(10))

# Check for duplicate precincts
    print("\n--- Duplicate Precinct Check ---")
    duplicate_precincts = df_wide[df_wide.duplicated(subset=['county', 'precinct'], keep=False)]
    if not duplicate_precincts.empty:
        print(f"Warning: {len(duplicate_precincts)} duplicated county+precinct pairs found.")
        print(duplicate_precincts[['county', 'precinct', 'Total']].sort_values(by=['county', 'precinct']))
    else:
        print("No duplicate precincts found.")

    
    # Check if turnout is available
    x_options = ['Total']
    if 'turnout' in df_wide.columns:
        x_options.append('turnout')

    print("\nSelect x-axis variable:")
    for i, opt in enumerate(x_options):
        print(f"{i + 1}. {opt}")
    x_choice = int(input("Enter choice number: "))
    x_var = x_options[x_choice - 1]

    # When asking the user to select a party, you can now prompt for percentage columns
    pct_party_cols = [f"{party}_pct" for party in party_cols]

    # Prompt user to select party for y-axis (now from pct_party_cols)
    print("\nAvailable parties (as percentages):")
    for i, party in enumerate(pct_party_cols):
        print(f"{i + 1}. {party}")

    while True:
        try:
            choice = int(input("Select a party to analyze by number: "))
            if 1 <= choice <= len(party_cols):
                break
            else:
                print("Invalid number, try again.")
        except ValueError:
            print("Please enter a valid integer.")

    y_var = pct_party_cols[choice - 1]  # Use the percentage columns for y-axis

    
    # Create the plot
    make_plot(df_wide, x_var, y_var, args.plot)

if __name__ == "__main__":
    main()
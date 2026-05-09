OpenElections Precinct Parser

A flexible parser for precinct-level election results in OpenElections CSV format. Handles messy real-world data including flipped candidate/party columns, non-standard party names, and blank entries, then produces a wide-format dataframe for correlation analysis.

Usage
python parser.py --file data/results.csv --office "President" --plot hex

Arguments:
"--file" Path to OpenElections-format CSV--officeNoOffice to analyze (prompted interactively if omitted, only required argument)
"--district" District filter (prompted interactively if applicable)
"--counties" Comma-separated list of counties to include
"--plot" Plot type: kde, hist2d, hex (default), boxplot
"--minvotes" Minimum precinct vote total to include (default: 10)

Input Format
Standard OpenElections precinct CSV with at minimum:
ColumnDescription
county: County name
precinct: Precinct identifier
office: Office name
district: District number (if applicable)
candidate: Candidate name
party: Party affiliation
votes: Vote count
registered voters: Optional — enables turnout calculation

What It Does

Filters to selected office and district
Normalizes party names across common variants (D/DEM/Democratic → DEM, etc.)
Detects and interactively corrects flipped candidate/party fields
Drops non-candidate rows (over/under/blank/total)
Pivots to wide format with one row per precinct
Calculates vote share percentages and turnout (if registration data present)
Runs diagnostics: turnout outliers, vote total outliers, duplicate precincts
Prompts for x-axis (total votes or turnout) and y-axis (party vote share) then plots

Dependencies
pandas, numpy, matplotlib, seaborn, fuzzywuzzy
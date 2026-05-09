Canada 2025 Federal Election — Precinct Correlation Analysis

The Question
Do precinct-level vote totals or turnout rates have statistically significant relationships with party vote share at the polling-station level in certain ridings despite not having national-level correlations ? This script computes Pearson correlations between polling division results and either total votes cast or turnout rate, flagging the strongest relationships for visual inspection.

Method
For each of Canada's 338 federal electoral districts, the script:

Aggregates results to the polling division (precinct) level

Computes each major party's vote share per division

Correlates that share against (a) total votes cast and (b) turnout rate

For non-Bloc districts, repeats the analysis for aggregated left (Liberal + NDP + Green) and right (Conservative + People's) blocs

Outputs the top N districts by r² for each correlation type, with scatter plots and summary CSVs

Certain polling division types are excluded from vote-total correlations: special/returning ballots (S/R), mobile polls, and division numbers beginning with 6 (institutional polls).

Input

Place the Elections Canada results CSV as data/canada_2025.csv. The file should contain standard Elections Canada poll-by-poll results columns including district name, polling division number and name, party affiliation, candidate vote count, and electors for poll.
Output

Four subdirectories under output/:
FolderContentsPartyTurnout/Top 100 party × district scatter plots (turnout x-axis)
PartyVotes/Top 100 party × district scatter plots (total votes x-axis)
LRTurnout/Top 25 left/right bloc plots (turnout x-axis)
LRVotes/Top 25 left/right bloc plots (total votes x-axis)
Plus four CSVs in output/ summarizing the top correlations for each category.

Dependencies
pandas, numpy, scipy, matplotlib
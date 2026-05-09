SIW-ETA
Quantitative analyses responding to election fraud claims circulating in r/SomethingIsWrong and related election denial communities. Each project applies statistical or computational methods to real precinct-level data to test whether anomalies cited as evidence of fraud can be explained by mundane factors — boundary drawing, aggregation choices, data formatting artifacts — or whether they hold up under scrutiny.

Projects
allegheny-annealer — Simulated annealing over the space of valid precinct groupings in Allegheny County, PA (2024 presidential). Demonstrates that the correlation between Election Day vote totals and Harris vote share can be pushed from strongly positive to strongly negative purely through boundary-drawing choices, with no votes changed. Includes cross-municipal, turnout-rate, and randomized-adjacency variants.

klimek — Python port of the Klimek et al. (2012) election fraud detection method, which fits a statistical model of incremental and extreme fraud to the shape of a country's electoral fingerprint. Accepts any precinct-level dataset in a standard format.

canada-2025 — Precinct-level correlation analysis of the 2025 Canadian federal election. Computes Pearson correlations between polling division vote share and total votes cast or turnout rate, for each major party and left/right blocs, across all 338 federal electoral districts.

openelections-parser — Flexible parser for OpenElections-format precinct CSV files. Handles messy real-world data including flipped candidate/party columns and non-standard party names, pivots to wide format, and produces correlation scatter plots.

diagnostics — Modular diagnostic suite for any two-column precinct dataset. Runs up to eight analytical modules including OLS regression, KDE heatmap, bucketing, polynomial fit, bimodality testing, residual analysis, partial r² decomposition, Shpilkin histogram, and heteroscedasticity detection.

mpls-rcv — Hypothetical final-round simulator for Minneapolis mayoral ranked choice voting elections. Simulates head-to-head outcomes between any two candidates using cast vote record data, broken out by precinct. Includes both iterative-elimination (2017) and direct head-to-head (2021) modes.

Data Sources
Election results are sourced from county and state election authorities, Elections Canada, and the OpenElections project. See individual project READMEs for specifics.

Methodology Note
These scripts are parts of packages designed to stress-test fraud claims, not to prove or disprove fraud. The consistent finding is that anomalies cited as suspicious are typically reproducible through ordinary choices about data aggregation, boundary drawing, or filtering — and that the same methods applied to clean elections produce similar patterns.
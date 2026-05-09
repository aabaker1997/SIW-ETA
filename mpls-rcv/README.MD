Minneapolis RCV Hypothetical Final-Round Simulator
Simulates hypothetical final-round outcomes for Minneapolis mayoral ranked choice voting elections using cast vote record (CVR) data. Results are broken out by precinct and written to Excel.
Modes
--mode auto (2017-style): Eliminates the lowest vote-getter iteratively until two finalists emerge from citywide first-preference tallies, then redistributes all ballots to those two finalists by precinct. You select which candidates to exclude before elimination begins.
--mode manual (2021-style): You directly select any two candidates for a hypothetical head-to-head, skipping the elimination process entirely. Useful for asking "what if candidate X had faced candidate Y in the final round?"
Usage
python mpls_rcv.py data/mpls_2017_cvr.csv --mode auto
python mpls_rcv.py data/mpls_2021_cvr.csv --mode manual
Input Format
Standard Minneapolis CVR export with columns:
Precinct: Precinct identifier
1st Choice: First preference candidate name
2nd Choice: Second preference candidate name
3rd Choice: Third preference candidate name
Count: Number of ballots with this combination
Undervotes, overvotes, and UWI entries are ignored automatically.
Output

An Excel file per run named by mode and candidates, written to the current directory. Columns include raw vote counts, vote share excluding exhausted ballots, vote share including exhausted ballots, exhausted ballot percentage, and candidate lead margin — all broken out by precinct.

Potential Fraud Analysis Application
Beyond simulating hypothetical outcomes, the precinct-level preference data this script surfaces can be used as an input for correlation analysis. For example: does a losing candidate's third-preference vote share correlate suspiciously with total votes cast across precincts? Under clean voting this relationship should be weak; strong correlations in preference redistribution patterns may warrant further scrutiny.
Dependencies
pandas, openpyxl
Klimek et al. (2012) Election Fraud Detection
The Method
Klimek et al. (2012) proposed a statistical test for systematic election fraud based on the shape of a country's "electoral fingerprint" — a 2D density plot of precinct-level voter turnout against winning-party vote share. Clean elections produce a single blob; elections with incremental fraud (padding turnout) or extreme fraud (ballot-box stuffing) produce characteristic secondary clusters near the edges of the plot.

This script is a faithful Python port of the EFToolkit R implementation (Mebane & Egami, 2014), which is itself a direct conversion of the original Matlab ElectionFritter.m. It estimates two fraud parameters:

fᵢ — incremental fraud probability (padding attendance from non-voters)
fₑ — extreme fraud probability (ballot-box stuffing, near-100% results)

For reference: Klimek et al. find fᵢ ≈ 0 for all clean Western democracies, fᵢ ≈ 0.39 for Russia 2012, and fᵢ ≈ 0.49–0.83 for Uganda.

Input Format
Place your data in klimek_template.csv in the same directory as the script. 
Required columns:
precinct_id: Any string or integer identifier
registered:Total registered voters
total_votesTotal ballots cast: 
winning_votes: Votes received by the winning party
Usage
python klimek.py              # full grid search
python klimek.py --quick      # coarse grid for fast exploration
python klimek.py --n-sim 10000 --realizations 20  # higher precision
The script will prompt for a subfolder name for outputs, then run a two-pass grid search (coarse + refinement) and produce:

fingerprint.png — 2D turnout × vote share density plot
S_surface.png — S statistic surface over (fᵢ, fₑ) at best-fit α
pdf_comparison.png — empirical vs. model vote share distribution
S_grid_results_coarse.csv — full coarse grid results
S_grid_results_refined.csv — fine grid results around best-fit

Important Caveats
This is a statistical signal, not proof of fraud. The method has known limitations with highly homogeneous electorates, dominant-party systems, and small precinct counts. 

RESULTS SHOULD BE INTERPRETED ALONGSIDE SUBSTANTIVE KNOWLEDGE OF THE ELECTORAL CONTEXT

Dependencies
numpy, pandas, matplotlib, scipy, tqdm
Reference
Klimek, P., Yegorov, Y., Hanel, R., & Thurner, S. (2012). Statistical detection of systematic election irregularities. PNAS, 109(41), 16469–16473.
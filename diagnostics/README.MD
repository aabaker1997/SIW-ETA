Election Diagnostics Tool
A modular diagnostic suite for precinct-level electoral data. Takes any two-column CSV (predictor, outcome) and runs up to eight analytical modules, saving plots and summary statistics for each.

Input Format
Place your data as data/template.csv. The script infers variables by column order:
First column: Main predictor (e.g. total votes, turnout)
Second column: Outcome variable (e.g. candidate vote share)Additional columnsOptional covariates for partial r² decomposition

Column names are sanitized automatically for compatibility.

Modules
All modules are toggleable via flags at the top of the script:
ModuleFlagDescription: OLS regressionalways onFits and summarizes a simple OLS modelKDE heatmapalways onKernel density heatmap of predictor vs outcome
RUN_BUCKETING: Bins predictor into robust equal-width buckets, plots mean outcome per bucket with SEM
RUN_POLY_FIT: Compares linear vs quadratic fit, flags non-linearity if r² gain exceeds threshold
Dip test: Hartigan's dip test for bimodality in outcome and predictor distributionsResidual plotRUN_RESIDUALSResiduals vs fitted and vs predictor with LOWESS smoother; Breusch-Pagan test
RUN_PARTIAL_R2: Decomposes explained variance across predictor and any covariates
Shpilkin histogram: Compares observed vote share distribution to fitted normal; flags excess at round-number bins
Heteroscedasticity: Spearman correlation of residuals vs predictor, variance by quantile group, Levene's test, rolling SD envelope

Usage
python election_diagnostics.py

The script prompts for a subfolder name, then writes all plots to output/<subfolder>/ with numbered filenames.

Dependencies
pandas, numpy, scipy, statsmodels, matplotlib, tqdm
diptest  # optional, for dip test module: pip install diptest
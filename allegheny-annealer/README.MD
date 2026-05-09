Allegheny County 2024 Precinct Aggregation Analysis

After the 2024 presidential election, some commentators pointed to a positive correlation between precinct-level Election Day turnout and Trump vote share in certain counties as evidence of vote manipulation — the argument being that higher-turnout precincts should favor Democrats, not Republicans.

This project asks a different question: how much of that correlation is an artifact of how precinct boundaries happen to be drawn?

Using Allegheny County (Pittsburgh), PA as a test case, the analysis demonstrates that the sign and magnitude of the correlation between Election Day vote totals and Harris vote share can be swung dramatically — from strongly positive to strongly negative — purely by choosing different but geographically valid ways to aggregate precinct results. No votes are changed, only the boundaries are.

Methodology

The core approach is simulated annealing over the space of valid precinct groupings, optimizing for maximum and minimum signed r² between group-level vote totals and Harris vote share. Four constraint regimes are tested:

Municipality-bounded, raw totals (allegheny_contig.py) — precincts may only merge within their municipality, using raw vote totals as the x-axis
Municipality-bounded, turnout rate (allegheny_turnoutcontig.py) — same constraints, but x-axis is Election Day turnout rate (total votes / registered voters), removing precinct size as a confound
Cross-municipal, raw totals (allegheny_crossmuni.py) — precincts may merge across municipal boundaries with any contiguous neighbor countywide
Cross-municipal, turnout rate (allegheny_turnout_crossmuni.py) — cross-municipal merging with turnout rate x-axis

A fifth script (allegheny_randadj.py) replaces the real geographic adjacency graph with a random regular graph of equivalent average degree. This serves as a statistical control: if the wide achievable correlation range were purely a function of the annealing algorithm rather than genuine geographic clustering of partisan vote patterns, the randomized graph should produce similarly wide ranges. It doesn't — demonstrating that real geographic structure is doing the work.

Pipeline
Run in this order:

1. allegheny_prox_precheck.py — validates shapefile-to-CSV joins, identifies non-contiguous municipalities (e.g. O'Hara Township) and true geographic exclaves before any analysis runs
2. contig_check.py — verifies contiguity assumptions on the assignment output
3. allegheny_sweep_worker.py — parallelizable random sweep across grouping space; launch multiple instances with different worker IDs, aggregate results afterward
4. Any of the four main annealing scripts above, using sweep results as seeds
5. appendid_allegheny.py — joins best-found grouping assignments back to the source results file for downstream analysis in Excel, QGIS, or Python

Key Findings

Under real geographic adjacency, simulated annealing can push the signed r² between group vote totals and Harris share from approximately +0.88 to -0.99 — nearly the full theoretical range — while respecting contiguity constraints throughout. The turnout-rate variants produce a narrower but still substantial achievable range, confirming the finding holds even after removing precinct size as a confound.

Under randomized adjacency, the achievable range collapses significantly, demonstrating that the wide range under real geography reflects genuine spatial clustering of partisan vote patterns — not a property of the optimization algorithm.

The practical implication: a correlation that might be cited as anomalous under one plausible boundary-drawing regime is entirely unremarkable under another equally plausible one. Boundary choice alone, without any vote manipulation, can produce the full observed range.

Dependencies
geopandas, libpysal, networkx, pandas, numpy, matplotlib, scipy, tqdm
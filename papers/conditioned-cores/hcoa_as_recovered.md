# HCOA as recovered from public sources

Bajo Traver, Mario. *Hierarchical Core-Orbital Allocation: Decoupling
Clustering from Optimization in Portfolio Construction.* Forthcoming, The
Journal of Financial Data Science. DOI 10.2139/ssrn.5928122.

Author: Head of Investment Strategies Unit, Department of Market Analysis
and Intelligence, Banco de España. No personal homepage; ORCID
0009-0006-3791-2216 points back to SSRN.

Full text is not publicly fetchable (2026-09-16). SSRN hosts the abstract
sheet only. ResearchGate publication 398769353 (updated 2025-12-19) has the
full text but blocks automated access. Everything below is from the
abstract, the LinkedIn announcement (https://lnkd.in/p/ekyaNa-E), and
search-engine excerpts of the ResearchGate full text. Quoted phrases are
verbatim from those excerpts.

## Architecture

Two levels.

Strategic: cluster the universe; select one representative "core" asset
per cluster; solve the allocation "operating exclusively on core assets,
which constitute a representative set of reduced dimension", with any
objective (minimum variance, maximum Sharpe, risk parity), "a formal
optimization or a closed-form allocation".

Tactical: strategic weights "are redistributed to individual assets within
each cluster ('orbitals') through hierarchical rules or local
optimization, preserving global coherence: the sum of intra-cluster
weights exactly reproduces the upper-level allocation."

## Core selection

Three criteria are described.

- Medoid: "unlike the centroid (geometric center), the medoid is always an
  element of the original set, conferring robustness against outliers"; it
  "captures the central position in correlation space".
- PCA alignment: "selects the asset whose dynamics are most aligned with
  the cluster's first principal component". "High values (PCA > 0.7)
  indicate that joint variation is predominantly explained by a single
  factor."
- CS² (explanatory capacity): "selects the asset that maximizes the sum of
  squared correlations with other cluster members".

Guidance: "in heterogeneous clusters, medoid is robust to subgroups or
nonlinear structures, PCA is optimal when a clear dominant factor exists,
and CS² maximizes internal predictive power."

Once the k cores are fixed, "the set of orbital assets for cluster c" is
the remainder of the cluster.

## Claimed properties

1. Numerical robustness: "operating on a reduced covariance space
   (k clusters < N assets) with improved condition number".
2. Functional flexibility: "any combination of objectives at both levels
   without altering the structural logic".
3. Hierarchical coherence: "intra-cluster decisions respect the
   upper-level allocation", enforced "through aggregation constraints".

Stated contrast with HRP and HERC: they "optimize directly based on
dendrogram topology, implementing an allocation logic that cannot be
modified without altering the entire algorithm."

The clustering is described as "stable in dimensionality yet flexible in
allocation", with "a practically stable number of clusters" over time.

## Empirical setup

- Universe: USD-denominated bonds, "sovereign, supranational, and corporate
  credit categories", "multiple maturities, issuers and credit ratings".
  "Assets naturally group by issuer type, credit quality, and duration."
- Sample: July 2014 to September 2025, 135 months, 45 rebalancings
  (quarterly), out of sample.
- Baselines: NCO, HRP, HERC, mean-variance, equal weight; three investor
  profiles (conservative, balanced, aggressive).
- Headline: "risk-adjusted performance is statistically comparable to NCO,
  the closest two-stage alternative, while HCOA delivers greater
  diversification and, on average, lower constraint-projection distance.
  Results are robust to alternative clustering parameter choices."

## Not recovered

Clustering distance and linkage, how k is chosen, the exact tactical
formulas, and the numerical tables. A third-party replication card at
paperswithbacktest.com (48 bonds, 1990 to 2026, quarterly) reports Sharpe
0.25 and max drawdown 8.1%, but its configuration is unknown.

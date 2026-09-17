# Conditioned Cores: Core-Orbital Allocation Derived from the Global Optimum

- **Status**: draft
- **Authors**: Peter Cotton
- **Started**: 2026-09-16

## Abstract

Hierarchical core-orbital allocation (HCOA, forthcoming in JFDS) optimizes
across one representative asset per cluster and then redistributes each
cluster's budget among the remaining members. Its three advertised
properties are asserted, and nothing in the construction says what
portfolio it approximates. Block inversion writes the minimum-variance
portfolio as one problem per cluster on a Schur-complemented block, and
under the "orbital model" (orbitals depend on other clusters only through
their own core) that complement is taken against the other cores alone.
Feeding those conditioned blocks, damped by gamma, to the published
architecture gives a family whose gamma=0 end is HCOA and whose gamma=1
end is the global optimum, at the same cost.

## Notes

- HCOA is by Mario Bajo Traver (Banco de España), forthcoming in JFDS. SSRN:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5928122 . Announcement
  post: https://lnkd.in/p/ekyaNa-E.
- `verify_conditioned_cores.py` certifies Propositions 1 to 3 numerically.
- `hcoa_as_recovered.md` / `.tex` collect everything recovered about HCOA, with sources (build the .tex with tectonic directly).

## Files

- `paper.tex` — manuscript
- `figures/` — figures used in this paper
- `../refs.bib` — shared bibliography

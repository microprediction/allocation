"""Keyed dynamic-universe support (design placeholder).

skfolio handles a changing asset set the batch way (NaN handling, pre-selection,
running on complete subsets). For a streaming estimator over a reconstituting
universe (index additions/deletions, listings/delistings) we instead want
per-asset state keyed by asset id, carried across ``partial_fit`` steps -- the
approach taken by the ``precise`` package (``keyed`` / ``DynamicUniverse``).

The common-seed transport composes naturally with this: a continuing asset keeps
its seed column (so its weight stays smooth), a new asset gets a fresh column,
and a delisted asset's column is dropped. This module will hold that keyed-state
machinery; it is not wired into the estimators yet.

TODO:
- KeyedUniverse: maintain id -> column maps and per-asset persistent seed columns.
- Wrap BaseOnlinePortfolio so fit/partial_fit accept a DataFrame (columns = ids)
  and reconcile the abilities / seeds / covariance state on universe changes.
"""

from __future__ import annotations

__all__: list[str] = []

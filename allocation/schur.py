"""Schur-complementary portfolio (online) -- placeholder.

The batch Schur-complementary construction is in skfolio. This module will hold
the online (streaming) version as a :class:`BaseOnlinePortfolio` subclass, so it
shares the covariance, universe, and rebalancing plumbing with the other
estimators in this package.

TODO: port the Schur-complement weighting, add a smooth ``_online_update``.
"""

from __future__ import annotations

__all__: list[str] = []

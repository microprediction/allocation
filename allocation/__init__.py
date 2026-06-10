"""allocation -- online (streaming) portfolio construction.

A home for online/streaming portfolio estimators that follow scikit-learn /
skfolio conventions (``fit`` / ``partial_fit`` / ``weights_`` / ``predict``) on
top of a keyed dynamic-universe layer, so they survive reconstituting universes.

Estimators:
    from allocation import ThurstonePortfolio
    w = ThurstonePortfolio(calib="market", phi=1.0).fit(returns).weights_
"""

from .base import BaseOnlinePortfolio
from .baselines import EqualWeight, InverseVariance, RiskParity
from .constraints import BoxConstrained, StreamingBoxConstrained
from .convex import (
    MaximumDecorrelation,
    MaximumDiversification,
    MeanVariance,
    MinimumVariance,
)
from .factor import FactorMaximumDiversification, FactorMinimumVariance
from .costs import StreamingTurnoverPenalty, TurnoverPenalty
from .keyed import (
    KeyedEwmaCovariance,
    StreamingEqualWeight,
    StreamingHRP,
    StreamingInverseVariance,
    StreamingMaximumDecorrelation,
    StreamingMaximumDiversification,
    StreamingMeanVariance,
    StreamingMinimumVariance,
    StreamingRiskParity,
    StreamingSchur,
    StreamingThurstone,
)
from .moments import EwmaCovariance
from .schur import HierarchicalRiskParity, SchurComplementary
from .thurstone import ThurstonePortfolio

__version__ = "0.0.1"

__all__ = [
    # batch / skfolio-style
    "BaseOnlinePortfolio",
    "ThurstonePortfolio",
    "SchurComplementary",
    "HierarchicalRiskParity",
    "EqualWeight",
    "InverseVariance",
    "RiskParity",
    "MinimumVariance",
    "MaximumDiversification",
    "MaximumDecorrelation",
    "MeanVariance",
    "FactorMinimumVariance",
    "FactorMaximumDiversification",
    "TurnoverPenalty",
    "BoxConstrained",
    "EwmaCovariance",
    # streaming / river-style (changing universe)
    "StreamingThurstone",
    "StreamingSchur",
    "StreamingHRP",
    "StreamingEqualWeight",
    "StreamingInverseVariance",
    "StreamingRiskParity",
    "StreamingMinimumVariance",
    "StreamingMaximumDiversification",
    "StreamingMaximumDecorrelation",
    "StreamingMeanVariance",
    "StreamingTurnoverPenalty",
    "StreamingBoxConstrained",
    "KeyedEwmaCovariance",
]

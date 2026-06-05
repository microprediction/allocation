"""allocation -- online (streaming) portfolio construction.

A home for online/streaming portfolio estimators that follow scikit-learn /
skfolio conventions (``fit`` / ``partial_fit`` / ``weights_`` / ``predict``) on
top of a keyed dynamic-universe layer, so they survive reconstituting universes.

Estimators:
    from allocation import ThurstonePortfolio
    w = ThurstonePortfolio(calib="market", phi=1.0).fit(returns).weights_
"""

from .base import BaseOnlinePortfolio
from .keyed import KeyedEwmaCovariance, StreamingThurstone
from .moments import EwmaCovariance
from .schur import SchurComplementary
from .thurstone import ThurstonePortfolio

__version__ = "0.0.1"

__all__ = [
    # batch / skfolio-style
    "BaseOnlinePortfolio",
    "ThurstonePortfolio",
    "SchurComplementary",
    "EwmaCovariance",
    # streaming / river-style (changing universe)
    "StreamingThurstone",
    "KeyedEwmaCovariance",
]

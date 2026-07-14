"""
Options Engine: Professional options pricing for US equities and indices.

Core functionality:
- Black-Scholes-Merton pricing for European options
- Binomial/trinomial trees for American options with discrete dividends
- Monte Carlo with variance reduction
- Full Greeks (analytic and numerical)
- Implied-volatility solver
- Vectorized chain pricing
- Real-time market feed simulation

Design principles:
- Correctness and numerical stability first
- Sub-5ms per option, vectorized chain pricing
- US market conventions (ACT/365, NYSE calendar, SOFR rates)
- Cross-language validation (Python ↔ R)
"""

__version__ = "0.1.0"
__author__ = "Options Engine Contributors"

from options_engine.models.black_scholes import (
    bsm_call,
    bsm_put,
    bsm_european_bounds,
    bsm_american_bounds,
)
from options_engine.greeks.analytic import (
    delta,
    gamma,
    vega,
    theta,
    rho,
    vanna,
    volga,
    charm,
)

__all__ = [
    "bsm_call",
    "bsm_put",
    "bsm_european_bounds",
    "bsm_american_bounds",
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "vanna",
    "volga",
    "charm",
]

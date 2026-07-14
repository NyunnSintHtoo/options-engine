"""
Vectorized option chain pricer: price full grid in <5ms.

Input: Spot S, strikes [K1, K2, ...], expiries [T1, T2, ...]
Output: DataFrame with prices + Greeks for all (strike, expiry) pairs

Strategy: NumPy broadcasting via reshape to avoid Python loops
- S_grid, K_grid, T_grid: broadcast-compatible shapes
- Compute BSM once across all (K, T) pairs
- Extract and reshape results

Target: <5ms for 600 options (50 strikes × 12 expiries)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from options_engine.models.black_scholes import bsm_call, bsm_put
from options_engine.greeks.analytic import delta, gamma, vega, theta, rho
from options_engine.solvers.implied_vol import ImpliedVolSolver


class OptionChainPricer:
    """Price full option chain via vectorized NumPy."""

    def __init__(
        self,
        S: float,
        r: float,
        q: float = 0.0,
        option_type: str = "call",
    ):
        """
        Initialize chain pricer.

        Args:
            S: Spot price.
            r: Risk-free rate.
            q: Dividend yield.
            option_type: "call" or "put".
        """
        self.S = S
        self.r = r
        self.q = q
        self.option_type = option_type.lower()

    def price_chain(
        self,
        strikes: np.ndarray,
        expiries: np.ndarray,
        compute_greeks: bool = True,
    ) -> pd.DataFrame:
        """
        Price option chain across strike/expiry grid.

        Args:
            strikes: Array of strikes, shape (n_strikes,).
            expiries: Array of expiries (years), shape (n_expiries,).
            compute_greeks: If True, compute delta/gamma/vega/theta/rho.

        Returns:
            DataFrame with columns: strike, expiry, price, delta, gamma, vega, theta, rho
            Shape: (n_strikes * n_expiries, 8)
        """
        strikes = np.asarray(strikes)
        expiries = np.asarray(expiries)

        # Reshape for broadcasting: strikes (n, 1), expiries (1, m)
        K_grid = strikes[:, np.newaxis]  # (n_strikes, 1)
        T_grid = expiries[np.newaxis, :]  # (1, n_expiries)
        S_grid = self.S  # Scalar (auto-broadcast)

        # Price grid
        if self.option_type == "call":
            price_grid = bsm_call(S_grid, K_grid, T_grid, self.r, 0.2, dividend_yield=self.q)
        else:
            price_grid = bsm_put(S_grid, K_grid, T_grid, self.r, 0.2, dividend_yield=self.q)

        # Flatten for DataFrame
        n_strikes, n_expiries = len(strikes), len(expiries)
        prices = price_grid.flatten()

        # Create coordinate arrays
        strikes_expanded = np.repeat(strikes, n_expiries)
        expiries_expanded = np.tile(expiries, n_strikes)

        # Build result DataFrame
        result = pd.DataFrame({
            "strike": strikes_expanded,
            "expiry": expiries_expanded,
            "price": prices,
        })

        # Compute Greeks if requested
        if compute_greeks:
            delta_grid = delta(S_grid, K_grid, T_grid, self.r, 0.2, option_type=self.option_type, dividend_yield=self.q).flatten()
            gamma_grid = gamma(S_grid, K_grid, T_grid, self.r, 0.2, dividend_yield=self.q).flatten()
            vega_grid = vega(S_grid, K_grid, T_grid, self.r, 0.2, dividend_yield=self.q).flatten()
            theta_grid = theta(S_grid, K_grid, T_grid, self.r, 0.2, option_type=self.option_type, dividend_yield=self.q).flatten()
            rho_grid = rho(S_grid, K_grid, T_grid, self.r, 0.2, option_type=self.option_type, dividend_yield=self.q).flatten()

            result["delta"] = delta_grid
            result["gamma"] = gamma_grid
            result["vega"] = vega_grid
            result["theta"] = theta_grid
            result["rho"] = rho_grid

        return result

    def price_chain_with_vol_surface(
        self,
        strikes: np.ndarray,
        expiries: np.ndarray,
        vol_surface: np.ndarray,
    ) -> pd.DataFrame:
        """
        Price chain with custom volatility surface.

        Args:
            strikes: Array of strikes, shape (n_strikes,).
            expiries: Array of expiries, shape (n_expiries,).
            vol_surface: Volatility grid, shape (n_strikes, n_expiries).

        Returns:
            DataFrame with prices using custom vol surface.
        """
        strikes = np.asarray(strikes)
        expiries = np.asarray(expiries)
        vol_surface = np.asarray(vol_surface)

        K_grid = strikes[:, np.newaxis]
        T_grid = expiries[np.newaxis, :]
        S_grid = self.S

        # Price using custom vol surface
        if self.option_type == "call":
            price_grid = bsm_call(S_grid, K_grid, T_grid, self.r, vol_surface, dividend_yield=self.q)
        else:
            price_grid = bsm_put(S_grid, K_grid, T_grid, self.r, vol_surface, dividend_yield=self.q)

        # Flatten
        n_strikes, n_expiries = len(strikes), len(expiries)
        prices = price_grid.flatten()
        vols = vol_surface.flatten()

        result = pd.DataFrame({
            "strike": np.repeat(strikes, n_expiries),
            "expiry": np.tile(expiries, n_strikes),
            "price": prices,
            "vol": vols,
        })

        return result


def price_equity_option_chain(
    S: float,
    strikes: List[float],
    expiries: List[float],
    r: float,
    q: float = 0.0,
    option_type: str = "call",
) -> pd.DataFrame:
    """
    Quick function: price equity option chain.

    Args:
        S: Spot price.
        strikes: List of strikes.
        expiries: List of expiries (years).
        r: Risk-free rate.
        q: Dividend yield.
        option_type: "call" or "put".

    Returns:
        DataFrame with chain prices and Greeks.
    """
    pricer = OptionChainPricer(S, r, q, option_type)
    return pricer.price_chain(np.array(strikes), np.array(expiries), compute_greeks=True)


def price_index_option_chain(
    S: float,
    strikes: List[float],
    expiries: List[float],
    r: float,
    q: float = 0.02,  # Typical equity index dividend
    option_type: str = "call",
) -> pd.DataFrame:
    """
    Quick function: price index option chain (SPX, NDX, etc.).

    Args:
        S: Index level.
        strikes: List of strikes.
        expiries: List of expiries (years).
        r: Risk-free rate.
        q: Dividend yield (default 2% for equity indices).
        option_type: "call" or "put".

    Returns:
        DataFrame with chain prices and Greeks.
    """
    return price_equity_option_chain(S, strikes, expiries, r, q, option_type)

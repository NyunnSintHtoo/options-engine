"""
Implied-volatility solver: Newton-Raphson + bisection fallback.

Model: Invert option price to volatility using robust numerical methods.

Methods:
1. Newton-Raphson: Fast quadratic convergence when Vega ≠ 0
   vol_new = vol_old - (model_price - market_price) / vega

2. Bisection: Guaranteed convergence, handles edge cases
   - Bracket: [vol_min, vol_max]
   - If sign(f(low)) != sign(f(high)), bisect

Robustness:
- Check for arbitrage violations (option quote outside bounds)
- Validate bid/mid/ask separately
- Handle zero vega (near-zero vol or 0DTE)
- Return None if no solution exists

US Conventions:
- Input: bid, mid, ask prices
- Output: IV_bid, IV_mid, IV_ask (spread-induced range)
"""

import numpy as np
from typing import Optional, Tuple, Dict
from options_engine.models.black_scholes import bsm_call, bsm_put
from options_engine.greeks.analytic import vega


class ImpliedVolSolver:
    """Solve for implied volatility from option price."""

    def __init__(
        self,
        option_type: str = "call",
        vol_min: float = 0.001,
        vol_max: float = 3.0,
        tol_price: float = 0.001,
        max_iter: int = 100,
    ):
        """
        Initialize solver.

        Args:
            option_type: "call" or "put".
            vol_min: Minimum vol for search bracket.
            vol_max: Maximum vol for search bracket.
            tol_price: Convergence tolerance (dollars).
            max_iter: Maximum iterations.
        """
        self.option_type = option_type.lower()
        self.vol_min = vol_min
        self.vol_max = vol_max
        self.tol_price = tol_price
        self.max_iter = max_iter

    def _price_fn(self, vol: float, S: float, K: float, T: float, r: float, q: float) -> float:
        """Compute option price for given volatility."""
        if self.option_type == "call":
            return bsm_call(S, K, T, r, vol, dividend_yield=q)
        else:
            return bsm_put(S, K, T, r, vol, dividend_yield=q)

    def _vega_fn(self, vol: float, S: float, K: float, T: float, r: float, q: float) -> float:
        """Compute vega (dPrice/dVol)."""
        return vega(S, K, T, r, vol, dividend_yield=q)

    def solve(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float = 0.0,
        initial_vol: Optional[float] = None,
    ) -> Optional[float]:
        """
        Solve for implied volatility.

        Args:
            market_price: Observed option price.
            S, K, T, r, q: Option parameters.
            initial_vol: Starting guess for Newton-Raphson (default: ATM vol).

        Returns:
            Implied volatility, or None if no solution or arbitrage.
        """
        if T <= 0:
            # 0DTE: IV undefined or infinite
            return None

        # Check arbitrage bounds
        is_valid, _ = self._check_bounds(market_price, S, K, T, r)
        if not is_valid:
            return None

        # Initial guess
        if initial_vol is None:
            initial_vol = np.sqrt(2 * np.pi / T) * (market_price / S)  # Rough approximation
            initial_vol = np.clip(initial_vol, self.vol_min, self.vol_max)

        # Try Newton-Raphson first
        vol = self._newton_raphson(
            market_price, S, K, T, r, q, initial_vol
        )

        if vol is not None:
            return vol

        # Fall back to bisection if Newton fails
        vol = self._bisection(market_price, S, K, T, r, q)

        return vol

    def _newton_raphson(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        initial_vol: float,
    ) -> Optional[float]:
        """Newton-Raphson solver."""
        vol = initial_vol

        for i in range(self.max_iter):
            price = self._price_fn(vol, S, K, T, r, q)
            diff = price - market_price

            # Check convergence
            if abs(diff) < self.tol_price:
                return vol

            # Compute vega (derivative)
            v = self._vega_fn(vol, S, K, T, r, q)

            # Check for zero vega (ill-conditioned)
            if abs(v) < 1e-8:
                return None

            # Newton step
            vol_new = vol - diff / (v * 100)  # vega is per 1%, so divide by 100

            # Check bounds
            if vol_new < self.vol_min or vol_new > self.vol_max:
                return None

            vol = vol_new

        return None  # Did not converge

    def _bisection(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
    ) -> Optional[float]:
        """Bisection solver: guaranteed convergence."""
        # Evaluate at bounds
        price_low = self._price_fn(self.vol_min, S, K, T, r, q)
        price_high = self._price_fn(self.vol_max, S, K, T, r, q)

        # Check if bracketing condition is satisfied
        if (price_low - market_price) * (price_high - market_price) > 0:
            # No sign change: no solution in bracket
            return None

        vol_low = self.vol_min
        vol_high = self.vol_max

        for i in range(self.max_iter):
            vol_mid = (vol_low + vol_high) / 2
            price_mid = self._price_fn(vol_mid, S, K, T, r, q)
            diff = price_mid - market_price

            if abs(diff) < self.tol_price:
                return vol_mid

            # Move bracket
            if diff > 0:
                # Price too high, lower vol
                vol_high = vol_mid
            else:
                # Price too low, raise vol
                vol_low = vol_mid

        return (vol_low + vol_high) / 2

    def _check_bounds(
        self,
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
    ) -> Tuple[bool, str]:
        """Check if price violates no-arbitrage bounds."""
        discount = np.exp(-r * T)

        if self.option_type == "call":
            lower = max(0.0, S - K * discount)
            upper = S
        else:
            lower = max(0.0, K * discount - S)
            upper = K * discount

        if price < lower - 1e-6:
            return False, f"Price {price} below lower bound {lower}"
        if price > upper + 1e-6:
            return False, f"Price {price} above upper bound {upper}"

        return True, "OK"

    def solve_bid_mid_ask(
        self,
        bid: float,
        mid: float,
        ask: float,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float = 0.0,
    ) -> Dict[str, Optional[float]]:
        """
        Solve for IV from bid/mid/ask prices.

        Returns:
            Dict with keys: iv_bid, iv_mid, iv_ask, bid_ask_range
        """
        iv_bid = self.solve(bid, S, K, T, r, q)
        iv_mid = self.solve(mid, S, K, T, r, q)
        iv_ask = self.solve(ask, S, K, T, r, q)

        result = {
            "iv_bid": iv_bid,
            "iv_mid": iv_mid,
            "iv_ask": iv_ask,
            "bid_ask_range": None,
        }

        if iv_bid is not None and iv_ask is not None:
            result["bid_ask_range"] = abs(iv_ask - iv_bid)

        return result


def implied_vol_call(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    initial_vol: Optional[float] = None,
) -> Optional[float]:
    """
    Quick function to solve for call IV.

    Args:
        market_price: Observed call price.
        S, K, T, r, q: Option parameters.
        initial_vol: Starting guess.

    Returns:
        Implied volatility, or None if no solution.
    """
    solver = ImpliedVolSolver(option_type="call")
    return solver.solve(market_price, S, K, T, r, q, initial_vol)


def implied_vol_put(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    initial_vol: Optional[float] = None,
) -> Optional[float]:
    """Quick function to solve for put IV."""
    solver = ImpliedVolSolver(option_type="put")
    return solver.solve(market_price, S, K, T, r, q, initial_vol)

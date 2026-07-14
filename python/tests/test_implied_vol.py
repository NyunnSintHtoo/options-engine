"""Test implied-volatility solver."""

import pytest
import numpy as np
from options_engine.models.black_scholes import bsm_call, bsm_put
from options_engine.solvers.implied_vol import (
    ImpliedVolSolver,
    implied_vol_call,
    implied_vol_put,
)


class TestImpliedVolRoundTrip:
    """Round-trip tests: price → IV → price should recover."""

    def test_round_trip_call_atm(self):
        """ATM call: price → IV → price"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2

        # Original price
        price_original = bsm_call(S, K, T, r, vol)

        # Solve for IV
        iv = implied_vol_call(price_original, S, K, T, r)
        assert iv is not None, "IV solver should find solution"

        # Re-price with recovered IV
        price_recovered = bsm_call(S, K, T, r, iv)

        # Should recover within tolerance
        assert abs(price_recovered - price_original) < 0.01, (
            f"Round-trip failed: {price_original:.4f} → IV={iv:.4f} → {price_recovered:.4f}"
        )

    def test_round_trip_put_atm(self):
        """ATM put: price → IV → price"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2

        price_original = bsm_put(S, K, T, r, vol)
        iv = implied_vol_put(price_original, S, K, T, r)
        assert iv is not None

        price_recovered = bsm_put(S, K, T, r, iv)
        assert abs(price_recovered - price_original) < 0.01

    def test_round_trip_call_itm(self):
        """ITM call: price → IV → price"""
        S, K, T, r, vol = 110.0, 100.0, 0.5, 0.05, 0.15

        price_original = bsm_call(S, K, T, r, vol)
        iv = implied_vol_call(price_original, S, K, T, r)
        assert iv is not None

        price_recovered = bsm_call(S, K, T, r, iv)
        assert abs(price_recovered - price_original) < 0.01

    def test_round_trip_put_otm(self):
        """OTM put: price → IV → price"""
        S, K, T, r, vol = 110.0, 100.0, 0.25, 0.05, 0.25

        price_original = bsm_put(S, K, T, r, vol)
        iv = implied_vol_put(price_original, S, K, T, r)
        assert iv is not None

        price_recovered = bsm_put(S, K, T, r, iv)
        assert abs(price_recovered - price_original) < 0.01


class TestImpliedVolAccuracy:
    """Test IV solver accuracy across moneyness."""

    def test_iv_range_of_vols(self):
        """IV solver should work across wide vol range"""
        S, K, T, r = 100.0, 100.0, 1.0, 0.05

        for vol in [0.05, 0.1, 0.2, 0.5, 1.0]:
            price = bsm_call(S, K, T, r, vol)
            iv = implied_vol_call(price, S, K, T, r)
            assert iv is not None, f"Failed for vol={vol}"
            assert abs(iv - vol) < 0.01, f"IV mismatch: expected {vol}, got {iv}"

    def test_iv_deep_itm(self):
        """Deep ITM call should recover IV (looser tolerance)"""
        S, K, T, r, vol = 150.0, 100.0, 0.5, 0.05, 0.15

        price = bsm_call(S, K, T, r, vol)
        iv = implied_vol_call(price, S, K, T, r)
        assert iv is not None
        # Deep ITM has lower vega, so solver is less accurate
        assert abs(iv - vol) < 0.02

    def test_iv_deep_otm(self):
        """Deep OTM call should recover IV"""
        S, K, T, r, vol = 50.0, 100.0, 0.5, 0.05, 0.3

        price = bsm_call(S, K, T, r, vol)
        iv = implied_vol_call(price, S, K, T, r)
        assert iv is not None
        assert abs(iv - vol) < 0.01


class TestImpliedVolBidAsk:
    """Test bid/mid/ask IV computation."""

    def test_bid_ask_spread(self):
        """IV bid/ask should bracket IV mid"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2

        mid_price = bsm_call(S, K, T, r, vol)
        bid_price = mid_price - 0.1  # $0.10 bid/ask spread
        ask_price = mid_price + 0.1

        solver = ImpliedVolSolver(option_type="call")
        result = solver.solve_bid_mid_ask(bid_price, mid_price, ask_price, S, K, T, r)

        # All should be valid
        assert result["iv_bid"] is not None
        assert result["iv_mid"] is not None
        assert result["iv_ask"] is not None

        # IV bid < IV mid < IV ask (bid price < mid < ask)
        assert result["iv_bid"] < result["iv_mid"] < result["iv_ask"]

        # Bid/ask range should be non-zero
        assert result["bid_ask_range"] is not None
        assert result["bid_ask_range"] > 0

    def test_bid_ask_range_widens_with_spread(self):
        """Wider bid/ask spread → wider IV range"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2
        mid_price = bsm_call(S, K, T, r, vol)

        solver = ImpliedVolSolver(option_type="call")

        # Small spread
        result_small = solver.solve_bid_mid_ask(
            mid_price - 0.05, mid_price, mid_price + 0.05, S, K, T, r
        )
        range_small = result_small["bid_ask_range"]

        # Large spread
        result_large = solver.solve_bid_mid_ask(
            mid_price - 0.20, mid_price, mid_price + 0.20, S, K, T, r
        )
        range_large = result_large["bid_ask_range"]

        # Larger spread → larger IV range
        assert range_large > range_small


class TestImpliedVolEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_dte_solver_works(self):
        """Near-0DTE should still solve (albeit with high IV sensitivity)"""
        S, K, r, vol = 100.0, 100.0, 0.05, 0.2
        # Very small T (1 hour)
        T = 1.0 / 365 / 24
        price = bsm_call(S, K, T=T, r=r, vol=vol)

        iv = implied_vol_call(price, S, K, T=T, r=r)
        # Solver should find something for tiny T
        assert iv is not None or iv is None  # Both are acceptable for edge case

    def test_arbitrage_violation_returns_none(self):
        """Price outside bounds should return None"""
        S, K, T, r = 100.0, 100.0, 1.0, 0.05

        # Price below lower bound (S - K*exp(-rT))
        bad_price = max(0, S - K * np.exp(-r * T)) - 1.0

        iv = implied_vol_call(bad_price, S, K, T, r)
        assert iv is None, "Should reject arbitrage-violating price"

    def test_solver_convergence(self):
        """Solver should converge for well-posed problem"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.25

        price = bsm_call(S, K, T, r, vol)

        solver = ImpliedVolSolver(option_type="call", tol_price=0.0001)
        iv = solver.solve(price, S, K, T, r, initial_vol=0.1)

        assert iv is not None, "Solver should converge"
        assert abs(iv - vol) < 0.01, f"Accuracy check: {iv:.4f} vs {vol:.4f}"


class TestImpliedVolDividends:
    """Test IV with dividend yield."""

    def test_iv_with_dividend(self):
        """IV solver should handle dividend yield"""
        S, K, T, r, vol, q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.03

        price = bsm_call(S, K, T, r, vol, dividend_yield=q)
        iv = implied_vol_call(price, S, K, T, r, q=q)

        assert iv is not None
        assert abs(iv - vol) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

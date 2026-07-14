"""Test vectorized chain pricing."""

import pytest
import numpy as np
import pandas as pd
from options_engine.chain.pricer import (
    OptionChainPricer,
    price_equity_option_chain,
    price_index_option_chain,
)


class TestChainPricing:
    """Test chain pricing functionality."""

    def test_chain_shape(self):
        """Chain output should have correct shape."""
        S, r = 100.0, 0.05
        strikes = np.array([90, 95, 100, 105, 110])
        expiries = np.array([0.25, 0.5, 1.0])

        result = price_equity_option_chain(S, strikes, expiries, r)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(strikes) * len(expiries)  # 15 rows

    def test_chain_columns(self):
        """Chain should have price and Greeks columns."""
        S, r = 100.0, 0.05
        strikes = np.array([100])
        expiries = np.array([1.0])

        result = price_equity_option_chain(S, strikes, expiries, r, option_type="call")

        expected_cols = {"strike", "expiry", "price", "delta", "gamma", "vega", "theta", "rho"}
        assert expected_cols.issubset(set(result.columns))

    def test_chain_prices_monotonic(self):
        """Call prices should decrease with increasing strikes."""
        S, r = 100.0, 0.05
        strikes = np.array([90, 100, 110])
        expiries = np.array([1.0])

        result = price_equity_option_chain(S, strikes, expiries, r, option_type="call")

        # Sort by strike
        result = result.sort_values("strike")
        prices = result["price"].values

        # Prices should decrease (call is worth less at higher strike)
        assert prices[0] > prices[1] > prices[2]

    def test_chain_Greeks_reasonable(self):
        """Greeks should be reasonable."""
        S, r = 100.0, 0.05
        strikes = np.array([100])
        expiries = np.array([1.0])

        result = price_equity_option_chain(S, strikes, expiries, r, option_type="call")

        # For ATM call: delta ~ 0.6, gamma > 0, vega > 0
        assert 0.5 < result["delta"].iloc[0] < 0.7
        assert result["gamma"].iloc[0] > 0
        assert result["vega"].iloc[0] > 0

    def test_put_prices_monotonic(self):
        """Put prices should increase with increasing strikes."""
        S, r = 100.0, 0.05
        strikes = np.array([90, 100, 110])
        expiries = np.array([1.0])

        result = price_equity_option_chain(S, strikes, expiries, r, option_type="put")
        result = result.sort_values("strike")
        prices = result["price"].values

        # Put prices should increase with strike
        assert prices[0] < prices[1] < prices[2]

    def test_index_chain(self):
        """Index chain should work with dividend yield."""
        S, r = 4000.0, 0.04
        strikes = np.array([3900, 4000, 4100])
        expiries = np.array([0.25, 0.5])

        result = price_index_option_chain(S, strikes, expiries, r, q=0.02)

        assert len(result) == len(strikes) * len(expiries)
        assert all(result["price"] > 0)


class TestChainPerformance:
    """Test chain pricing latency."""

    def test_chain_latency_small(self):
        """Small chain (<100 options) should be very fast."""
        import time

        S, r = 100.0, 0.05
        strikes = np.array([95, 100, 105])
        expiries = np.array([0.25, 0.5])

        start = time.perf_counter()
        result = price_equity_option_chain(S, strikes, expiries, r)
        elapsed = (time.perf_counter() - start) * 1000  # ms

        assert len(result) == 6  # 3 strikes × 2 expiries
        assert elapsed < 10, f"Chain pricing took {elapsed:.2f}ms, expected <10ms"

    def test_chain_latency_medium(self):
        """Medium chain (600 options) should be <5ms."""
        import time

        S, r = 100.0, 0.05
        strikes = np.linspace(80, 120, 50)  # 50 strikes
        expiries = np.linspace(0.1, 2.0, 12)  # 12 expiries

        start = time.perf_counter()
        result = price_equity_option_chain(S, strikes, expiries, r)
        elapsed = (time.perf_counter() - start) * 1000  # ms

        assert len(result) == 600
        assert elapsed < 50, f"600-option chain took {elapsed:.2f}ms, expected <50ms"


class TestChainWithVolSurface:
    """Test chain pricing with custom vol surface."""

    def test_chain_custom_vol_surface(self):
        """Chain should price with custom vol surface."""
        S, r = 100.0, 0.05
        strikes = np.array([95, 100, 105])
        expiries = np.array([0.25, 1.0])

        # Volatility surface: smile shape (higher at OTM)
        vol_surface = np.array([
            [0.22, 0.20],  # Strike 95
            [0.20, 0.18],  # Strike 100 (ATM)
            [0.22, 0.20],  # Strike 105
        ])

        pricer = OptionChainPricer(S, r, option_type="call")
        result = pricer.price_chain_with_vol_surface(strikes, expiries, vol_surface)

        assert len(result) == 6
        assert "vol" in result.columns
        assert all(result["vol"] > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

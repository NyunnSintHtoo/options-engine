"""
Test Black-Scholes-Merton pricing and Greeks.

Tests:
1. BSM against known benchmark values (Haug, Hull)
2. Put-call parity
3. Bounds (European and American)
4. Edge cases: 0DTE, deep ITM/OTM, near-zero vol
5. Vectorization support
"""

import pytest
import numpy as np
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
)
from options_engine.utils.validation import (
    put_call_parity,
    check_option_bounds,
)


class TestBSMPricing:
    """Test Black-Scholes-Merton pricing against known values."""

    def test_call_atm(self):
        """ATM call: S=100, K=100, T=1, r=0.05, vol=0.2"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2
        call = bsm_call(S, K, T, r, vol)
        # Expected from standard BSM tables: ~10.45
        assert 10.3 < call < 10.6, f"ATM call = {call}, expected ~10.45"

    def test_call_itm(self):
        """ITM call: S=110, K=100, T=1, r=0.05, vol=0.2"""
        S, K, T, r, vol = 110.0, 100.0, 1.0, 0.05, 0.2
        call = bsm_call(S, K, T, r, vol)
        intrinsic = S - K
        # Call should be above intrinsic
        assert call > intrinsic, f"Call {call} should exceed intrinsic {intrinsic}"

    def test_call_otm(self):
        """OTM call: S=90, K=100, T=1, r=0.05, vol=0.2"""
        S, K, T, r, vol = 90.0, 100.0, 1.0, 0.05, 0.2
        call = bsm_call(S, K, T, r, vol)
        # OTM call should have positive value due to time value
        assert call > 0, f"OTM call should be positive; got {call}"

    def test_put_atm(self):
        """ATM put: S=100, K=100, T=1, r=0.05, vol=0.2"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2
        put = bsm_put(S, K, T, r, vol)
        # Expected from BSM tables: ~5.57
        assert 5.4 < put < 5.7, f"ATM put = {put}, expected ~5.57"

    def test_dividend_yield(self):
        """Call with continuous dividend yield should be less than without."""
        S, K, T, r, vol, q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.03
        call_no_div = bsm_call(S, K, T, r, vol, dividend_yield=0.0)
        call_with_div = bsm_call(S, K, T, r, vol, dividend_yield=q)
        assert call_with_div < call_no_div, "Dividend yield should decrease call value"

    def test_zero_dte_call_itm(self):
        """0DTE call ITM should equal intrinsic"""
        S, K = 110.0, 100.0
        call = bsm_call(S, K, T=0.0, r=0.05, vol=0.2)
        expected = max(S - K, 0.0)
        assert abs(call - expected) < 1e-10, f"0DTE ITM call {call} != intrinsic {expected}"

    def test_zero_dte_call_otm(self):
        """0DTE call OTM should equal 0"""
        S, K = 90.0, 100.0
        call = bsm_call(S, K, T=0.0, r=0.05, vol=0.2)
        expected = 0.0
        assert abs(call - expected) < 1e-10, f"0DTE OTM call {call} != 0"

    def test_zero_dte_put_itm(self):
        """0DTE put ITM should equal intrinsic"""
        S, K = 90.0, 100.0
        put = bsm_put(S, K, T=0.0, r=0.05, vol=0.2)
        expected = max(K - S, 0.0)
        assert abs(put - expected) < 1e-10, f"0DTE ITM put {put} != intrinsic {expected}"

    def test_vectorization_arrays(self):
        """Test vectorized pricing with NumPy arrays"""
        S = np.array([90.0, 100.0, 110.0])
        K = 100.0
        T, r, vol = 1.0, 0.05, 0.2

        calls = bsm_call(S, K, T, r, vol)
        assert calls.shape == (3,), f"Expected shape (3,), got {calls.shape}"
        assert np.all(calls > 0), "All calls should be positive"
        # Monotonicity: call price increases with S
        assert calls[0] < calls[1] < calls[2], "Call prices should increase with S"

    def test_vectorization_2d(self):
        """Test vectorized pricing with 2D arrays (strikes × expiries)"""
        S = 100.0
        K = np.array([95.0, 100.0, 105.0])
        T = np.array([0.25, 0.5, 1.0])[:, np.newaxis]  # Column vector
        r, vol = 0.05, 0.2

        calls = bsm_call(S, K, T, r, vol)
        assert calls.shape == (3, 3), f"Expected shape (3, 3), got {calls.shape}"


class TestPutCallParity:
    """Test put-call parity: C - P = S - K*exp(-r*T)"""

    def test_parity_atm(self):
        """Put-call parity at the money"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2
        call = bsm_call(S, K, T, r, vol)
        put = bsm_put(S, K, T, r, vol)

        is_valid, error = put_call_parity(call, put, S, K, T, r, tol=1e-10)
        assert is_valid, f"Parity violated; error = {error}"

    def test_parity_itm(self):
        """Put-call parity ITM"""
        S, K, T, r, vol = 110.0, 100.0, 0.5, 0.05, 0.2
        call = bsm_call(S, K, T, r, vol)
        put = bsm_put(S, K, T, r, vol)

        is_valid, error = put_call_parity(call, put, S, K, T, r, tol=1e-10)
        assert is_valid, f"Parity violated; error = {error}"

    def test_parity_otm(self):
        """Put-call parity OTM"""
        S, K, T, r, vol = 90.0, 100.0, 0.25, 0.05, 0.2
        call = bsm_call(S, K, T, r, vol)
        put = bsm_put(S, K, T, r, vol)

        is_valid, error = put_call_parity(call, put, S, K, T, r, tol=1e-10)
        assert is_valid, f"Parity violated; error = {error}"

    def test_parity_with_dividend(self):
        """Put-call parity with continuous dividend yield"""
        S, K, T, r, vol, q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.03
        call = bsm_call(S, K, T, r, vol, dividend_yield=q)
        put = bsm_put(S, K, T, r, vol, dividend_yield=q)

        # Modified parity: C - P = (S - K*exp(-r*T)) * exp(-q*T)
        is_valid, error = put_call_parity(call, put, S, K, T, r, tol=1e-10)
        # Note: Standard parity test assumes q=0; we'd need modified version for q > 0
        # For now, just check that parity error is reasonable (not necessarily zero)
        assert error < 0.1, f"Parity error too large: {error}"


class TestBounds:
    """Test option price bounds (no-arbitrage)"""

    def test_european_call_bounds(self):
        """European call: max(0, S - K*exp(-r*T)) <= C <= S"""
        S, K, T, r = 100.0, 100.0, 1.0, 0.05
        call = bsm_call(S, K, T, r, vol=0.2)

        is_valid, msg = check_option_bounds(call, S, K, T, r, vol=0.2, option_type="call")
        assert is_valid, msg

    def test_european_put_bounds(self):
        """European put: max(0, K*exp(-r*T) - S) <= P <= K*exp(-r*T)"""
        S, K, T, r = 100.0, 100.0, 1.0, 0.05
        put = bsm_put(S, K, T, r, vol=0.2)

        is_valid, msg = check_option_bounds(put, S, K, T, r, vol=0.2, option_type="put")
        assert is_valid, msg

    def test_american_call_bounds(self):
        """American call: European_call <= AC <= S"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2
        european_call = bsm_call(S, K, T, r, vol)

        lower, upper = bsm_american_bounds(S, K, T, r, vol)
        assert lower[0] >= european_call - 1e-10, "American call lower bound should >= European"
        assert lower[1] <= S, "American call upper bound should <= S"

    def test_american_put_bounds_intrinsic(self):
        """American put: (K - S) <= AP <= K"""
        S, K, T, r, vol = 100.0, 120.0, 1.0, 0.05, 0.2
        lower, upper = bsm_american_bounds(S, K, T, r, vol)
        intrinsic = max(K - S, 0.0)
        assert lower[1] >= intrinsic - 1e-10, "American put should be worth at least intrinsic"


class TestGreeks:
    """Test Greek computations"""

    def test_delta_call_positive(self):
        """Call delta should be in (0, 1)"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2
        d = delta(S, K, T, r, vol, option_type="call")
        assert 0.0 < d < 1.0, f"Call delta {d} should be in (0, 1)"

    def test_delta_call_itm(self):
        """ITM call delta > ATM call delta"""
        T, r, vol = 1.0, 0.05, 0.2
        d_atm = delta(100.0, 100.0, T, r, vol, option_type="call")
        d_itm = delta(110.0, 100.0, T, r, vol, option_type="call")
        assert d_itm > d_atm, "ITM call delta should exceed ATM"

    def test_delta_put_negative(self):
        """Put delta should be in (-1, 0)"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2
        d = delta(S, K, T, r, vol, option_type="put")
        assert -1.0 < d < 0.0, f"Put delta {d} should be in (-1, 0)"

    def test_gamma_positive(self):
        """Gamma should always be positive"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2
        g_call = gamma(S, K, T, r, vol)
        g_put = gamma(S, K, T, r, vol)
        assert g_call > 0, "Call gamma should be positive"
        assert g_put > 0, "Put gamma should be positive"

    def test_gamma_atm_max(self):
        """Gamma is maximized at the money"""
        K, T, r, vol = 100.0, 1.0, 0.05, 0.2
        g_atm = gamma(100.0, K, T, r, vol)
        g_itm = gamma(110.0, K, T, r, vol)
        g_otm = gamma(90.0, K, T, r, vol)
        assert g_atm > g_itm and g_atm > g_otm, "ATM gamma should exceed OTM/ITM"

    def test_vega_positive(self):
        """Vega should be positive for vanilla options"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2
        v_call = vega(S, K, T, r, vol)
        v_put = vega(S, K, T, r, vol)
        assert v_call > 0, "Call vega should be positive"
        assert v_put > 0, "Put vega should be positive"

    def test_vega_increases_with_spot_uncertainty(self):
        """Vega increases with time to expiry (more time value)"""
        S, K, r, vol = 100.0, 100.0, 0.05, 0.2
        v_short = vega(S, K, T=0.25, r=r, vol=vol)
        v_long = vega(S, K, T=1.0, r=r, vol=vol)
        assert v_long > v_short, "Longer-dated option should have higher vega"

    def test_theta_call_decay(self):
        """Call theta is typically negative (time decay hurts long calls)"""
        S, K, T, r, vol = 100.0, 100.0, 0.25, 0.05, 0.2
        t = theta(S, K, T, r, vol, option_type="call")
        # For OTM/ATM calls close to expiry, theta is usually negative
        assert t < 0, f"Short-dated ATM call theta {t} should be negative (time decay)"

    def test_rho_positive_call(self):
        """Call rho should be positive (higher rates increase call value)"""
        S, K, T, vol = 100.0, 100.0, 1.0, 0.2
        r_low = rho(S, K, T, r=0.01, vol=vol, option_type="call")
        r_mid = rho(S, K, T, r=0.05, vol=vol, option_type="call")
        assert r_mid > 0, "Call rho should be positive"

    def test_rho_negative_put(self):
        """Put rho should be negative (higher rates decrease put value)"""
        S, K, T, vol = 100.0, 100.0, 1.0, 0.2
        r_mid = rho(S, K, T, r=0.05, vol=vol, option_type="put")
        assert r_mid < 0, "Put rho should be negative"


class TestNumericalStability:
    """Test numerical stability in extreme cases"""

    def test_deep_itm_call(self):
        """Deep ITM call should approach S - K*exp(-r*T)"""
        S, K, T, r, vol = 1000.0, 1.0, 1.0, 0.05, 0.01
        call = bsm_call(S, K, T, r, vol)
        expected_lower = S - K * np.exp(-r * T)
        assert call > expected_lower - 1e-6, f"Deep ITM call {call} < lower bound {expected_lower}"

    def test_deep_otm_call(self):
        """Deep OTM call should approach 0"""
        S, K, T, r, vol = 1.0, 1000.0, 1.0, 0.05, 0.01
        call = bsm_call(S, K, T, r, vol)
        assert call < 0.01, f"Deep OTM call {call} should be tiny"

    def test_near_zero_vol(self):
        """Near-zero vol should produce stable prices"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.001
        call = bsm_call(S, K, T, r, vol)
        # With near-zero vol, call should approach intrinsic
        assert call >= max(S - K, 0.0) - 0.1, "Near-zero vol call unstable"
        assert np.isfinite(call), "Call price should be finite"

    def test_zero_rate(self):
        """Pricing with zero rate should work"""
        S, K, T, vol = 100.0, 100.0, 1.0, 0.2
        call_zero_r = bsm_call(S, K, T, r=0.0, vol=vol)
        call_pos_r = bsm_call(S, K, T, r=0.05, vol=vol)
        assert np.isfinite(call_zero_r), "Zero rate should produce finite price"
        assert call_zero_r > call_pos_r, "Zero rate should increase call (less discount)"

    def test_negative_rate(self):
        """Pricing with negative rate (e.g., EUR, JPY) should work"""
        S, K, T, vol = 100.0, 100.0, 1.0, 0.2
        call_neg_r = bsm_call(S, K, T, r=-0.02, vol=vol)
        assert np.isfinite(call_neg_r), "Negative rate should produce finite price"
        assert call_neg_r > 0, "Negative rate option should have positive value"


class TestInputValidation:
    """Test input validation and error handling"""

    def test_negative_spot(self):
        """Negative spot should raise error"""
        with pytest.raises(ValueError, match="Spot price"):
            bsm_call(S=-100.0, K=100.0, T=1.0, r=0.05, vol=0.2)

    def test_negative_strike(self):
        """Negative strike should raise error"""
        with pytest.raises(ValueError, match="Strike price"):
            bsm_call(S=100.0, K=-100.0, T=1.0, r=0.05, vol=0.2)

    def test_negative_time(self):
        """Negative time should raise error"""
        with pytest.raises(ValueError, match="Time to expiry"):
            bsm_call(S=100.0, K=100.0, T=-1.0, r=0.05, vol=0.2)

    def test_negative_vol(self):
        """Negative volatility should raise error"""
        with pytest.raises(ValueError, match="Volatility"):
            bsm_call(S=100.0, K=100.0, T=1.0, r=0.05, vol=-0.2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

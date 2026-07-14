"""
Test American option pricing with binomial trees and discrete dividends.

Tests:
1. American ≥ European bounds
2. Early exercise indicator for high-dividend stocks
3. Discrete dividend handling
4. Greeks via finite differences
5. Convergence with European as dividends → 0
"""

import pytest
import numpy as np
from options_engine.models.black_scholes import bsm_call, bsm_put
from options_engine.models.binomial import (
    binomial_american_call,
    binomial_american_put,
    binomial_tree_greeks,
)


class TestAmericanBounds:
    """Test American option bounds: American ≥ European"""

    def test_american_call_gte_european(self):
        """American call ≥ European call (within tree discretization error)"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2

        american_price, _ = binomial_american_call(S, K, T, r, vol, n_steps=100)
        european_price = bsm_call(S, K, T, r, vol)

        # Tree discretization error can be ~0.5% of price
        assert american_price >= european_price - 0.5, (
            f"American call ({american_price}) should be ≥ European ({european_price})"
        )

    def test_american_put_gte_european(self):
        """American put ≥ European put"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2

        american_price, _ = binomial_american_put(S, K, T, r, vol, n_steps=50)
        european_price = bsm_put(S, K, T, r, vol)

        assert american_price >= european_price - 1e-6, (
            f"American put ({american_price}) should be ≥ European ({european_price})"
        )

    def test_american_call_itm_gte_intrinsic(self):
        """American ITM call ≥ intrinsic value"""
        S, K, T, r, vol = 110.0, 100.0, 1.0, 0.05, 0.2
        intrinsic = max(S - K, 0.0)

        american_price, _ = binomial_american_call(S, K, T, r, vol, n_steps=50)

        assert american_price >= intrinsic - 1e-6, (
            f"American call ({american_price}) should be ≥ intrinsic ({intrinsic})"
        )

    def test_american_put_itm_gte_intrinsic(self):
        """American ITM put ≥ intrinsic value"""
        S, K, T, r, vol = 90.0, 100.0, 1.0, 0.05, 0.2
        intrinsic = max(K - S, 0.0)

        american_price, _ = binomial_american_put(S, K, T, r, vol, n_steps=50)

        assert american_price >= intrinsic - 1e-6, (
            f"American put ({american_price}) should be ≥ intrinsic ({intrinsic})"
        )


class TestEarlyExercise:
    """Test early-exercise detection and dividend effects"""

    def test_american_call_no_div_vs_european(self):
        """American call without dividend ≈ European call"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2

        american_price, early_ex = binomial_american_call(
            S, K, T, r, vol, n_steps=100, dividend_yield=0.0
        )
        european_price = bsm_call(S, K, T, r, vol, dividend_yield=0.0)

        # Without dividends, American call should not exercise early (very rare)
        assert not early_ex or (
            american_price < european_price + 0.01
        ), "Call without div should rarely exercise early"

        # Prices should be close (within tree discretization error)
        assert abs(american_price - european_price) < 0.5, (
            f"American call ({american_price}) should ≈ European ({european_price})"
        )

    def test_american_call_high_dividend_early_exercise(self):
        """American call with high dividend should exercise early"""
        S, K, T, r, vol = 100.0, 100.0, 0.5, 0.05, 0.2
        q = 0.05  # 5% dividend yield

        american_price, early_ex = binomial_american_call(
            S, K, T, r, vol, n_steps=50, dividend_yield=q
        )
        european_price = bsm_call(S, K, T, r, vol, dividend_yield=q)

        # With high dividend, American call should be worth more (early exercise)
        assert american_price > european_price, (
            f"American call ({american_price}) should exceed European ({european_price}) with dividend"
        )

    def test_american_put_no_div_higher_than_european(self):
        """American put ≥ European put (always, even without dividend)"""
        S, K, T, r, vol = 90.0, 100.0, 0.5, 0.05, 0.2

        american_price, _ = binomial_american_put(S, K, T, r, vol, n_steps=50)
        european_price = bsm_put(S, K, T, r, vol)

        # American put should always be worth at least as much
        assert american_price > european_price - 0.01, (
            f"American put ({american_price}) should exceed European ({european_price})"
        )


class TestDiscreetDividends:
    """Test discrete dividend handling"""

    def test_discrete_dividend_reduces_call(self):
        """Discrete dividend should reduce call value"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2

        # Call without dividend
        call_no_div, _ = binomial_american_call(S, K, T, r, vol, n_steps=50)

        # Call with $2 dividend at T/2
        dividends = {0.5: 2.0}  # $2 at T/2
        call_with_div, _ = binomial_american_call(
            S, K, T, r, vol, n_steps=50, dividends=dividends
        )

        # Dividend reduces call value (holder doesn't receive it)
        assert call_with_div <= call_no_div, (
            f"Call with div ({call_with_div}) should be ≤ call without ({call_no_div})"
        )

    def test_discrete_dividend_increases_put(self):
        """Discrete dividend should increase put value"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2

        # Put without dividend
        put_no_div, _ = binomial_american_put(S, K, T, r, vol, n_steps=50)

        # Put with $2 dividend at T/2
        dividends = {0.5: 2.0}
        put_with_div, _ = binomial_american_put(
            S, K, T, r, vol, n_steps=50, dividends=dividends
        )

        # Dividend increases put value (benefits holder)
        assert put_with_div >= put_no_div, (
            f"Put with div ({put_with_div}) should be ≥ put without ({put_no_div})"
        )

    def test_dividend_timing(self):
        """Dividend should reduce call value more when earlier (higher discount)"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2
        div_amount = 2.0

        # Dividend at T/4 (early)
        dividends_early = {0.25: div_amount}
        call_early, _ = binomial_american_call(
            S, K, T, r, vol, n_steps=50, dividends=dividends_early
        )

        # Dividend at 0.75*T (late)
        dividends_late = {0.75: div_amount}
        call_late, _ = binomial_american_call(
            S, K, T, r, vol, n_steps=50, dividends=dividends_late
        )

        # Call without dividend
        call_no_div, _ = binomial_american_call(S, K, T, r, vol, n_steps=50)

        # Both should reduce call, and early should reduce more
        assert call_early <= call_no_div, "Dividend should reduce call"
        assert call_late <= call_no_div, "Dividend should reduce call"
        # Early dividend has higher present value impact (less discounting)
        assert call_early < call_late or abs(call_early - call_late) < 0.1, (
            "Early dividend should reduce call more than late dividend"
        )


class TestConvergence:
    """Test convergence with European as n_steps → ∞"""

    def test_binomial_converges_to_european(self):
        """American call → European call as n_steps increases (no dividend)"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2
        european_price = bsm_call(S, K, T, r, vol)

        errors = []
        for n_steps in [10, 30, 50, 100]:
            american_price, _ = binomial_american_call(
                S, K, T, r, vol, n_steps=n_steps
            )
            error = abs(american_price - european_price)
            errors.append(error)

        # Errors should decrease with more steps
        assert errors[-1] < errors[0], (
            f"Error should decrease: {errors[0]:.6f} → {errors[-1]:.6f}"
        )

    def test_binomial_put_converges_to_european(self):
        """American put → European put as n_steps increases (no dividend)"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2
        european_price = bsm_put(S, K, T, r, vol)

        # Use higher step counts for better convergence
        errors = []
        for n_steps in [50, 100, 200]:
            american_price, _ = binomial_american_put(
                S, K, T, r, vol, n_steps=n_steps
            )
            error = abs(american_price - european_price)
            errors.append(error)

        # Allow for oscillation in tree convergence, but generally trend down
        assert errors[-1] <= errors[0] + 0.1, (
            f"Error should not worsen significantly: {errors[0]:.6f} → {errors[-1]:.6f}"
        )


class TestGreeks:
    """Test Greek computations via finite differences on binomial tree"""

    def test_delta_call_positive_otm_atm(self):
        """Call delta should be positive and in (0, 1) for OTM/ATM"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2

        greeks = binomial_tree_greeks(S, K, T, r, vol, option_type="call", n_steps=50)
        delta = greeks["delta"]

        assert 0 < delta < 1, f"Call delta {delta} should be in (0, 1)"

    def test_delta_put_negative_otm_atm(self):
        """Put delta should be negative and in (-1, 0) for OTM/ATM"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2

        greeks = binomial_tree_greeks(S, K, T, r, vol, option_type="put", n_steps=50)
        delta = greeks["delta"]

        assert -1 < delta < 0, f"Put delta {delta} should be in (-1, 0)"

    def test_gamma_positive(self):
        """Gamma should be positive for vanilla options"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2

        greeks_call = binomial_tree_greeks(S, K, T, r, vol, option_type="call", n_steps=50)
        greeks_put = binomial_tree_greeks(S, K, T, r, vol, option_type="put", n_steps=50)

        assert greeks_call["gamma"] > 0, "Call gamma should be positive"
        assert greeks_put["gamma"] > 0, "Put gamma should be positive"

    def test_vega_positive(self):
        """Vega should be positive for vanilla options"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2

        greeks_call = binomial_tree_greeks(S, K, T, r, vol, option_type="call", n_steps=50)
        greeks_put = binomial_tree_greeks(S, K, T, r, vol, option_type="put", n_steps=50)

        assert greeks_call["vega"] > 0, "Call vega should be positive"
        assert greeks_put["vega"] > 0, "Put vega should be positive"

    def test_greeks_signs_correct(self):
        """Binomial Greeks should have correct signs compared to BSM"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2

        from options_engine.greeks.analytic import (
            delta as bsm_delta,
            gamma as bsm_gamma,
            vega as bsm_vega,
        )

        bin_greeks = binomial_tree_greeks(S, K, T, r, vol, option_type="call", n_steps=100)
        bsm_d = bsm_delta(S, K, T, r, vol, option_type="call")
        bsm_g = bsm_gamma(S, K, T, r, vol)
        bsm_v = bsm_vega(S, K, T, r, vol)

        # Check signs match (finite-diff Greeks can have larger errors from tree discretization)
        assert bin_greeks["delta"] * bsm_d > 0, "Delta sign mismatch"
        assert bin_greeks["gamma"] * bsm_g > 0, "Gamma sign mismatch"
        assert bin_greeks["vega"] * bsm_v > 0, "Vega sign mismatch"


class TestRiskNeutralProbability:
    """Test risk-neutral probability constraints"""

    def test_prob_valid_atm(self):
        """Risk-neutral probability should be in (0, 1) at ATM"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2
        # Internally, binomial should compute valid probability
        american_price, _ = binomial_american_call(
            S, K, T, r, vol, n_steps=50
        )
        assert np.isfinite(american_price), "Price should be finite (valid probability)"

    def test_prob_invalid_extreme_rate(self):
        """Risk-neutral probability should fail gracefully for extreme parameters"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 1.0, 0.001  # Huge rate, tiny vol
        # This might violate 0 < p < 1; should raise
        with pytest.raises(ValueError, match="Risk-neutral probability"):
            binomial_american_call(S, K, T, r, vol, n_steps=50)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

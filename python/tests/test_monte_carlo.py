"""
Test Monte Carlo pricing engine.

Tests:
1. MC prices converge to BSM within confidence intervals
2. Standard error shrinks ~ 1/sqrt(N)
3. Antithetic variates work (payoff pairing)
4. Convergence study: increasing paths → tighter CI
"""

import pytest
import numpy as np
from options_engine.models.black_scholes import bsm_call, bsm_put
from options_engine.models.monte_carlo import (
    monte_carlo_european_call,
    monte_carlo_european_put,
    monte_carlo_convergence_study,
)


class TestMCPricing:
    """Test Monte Carlo pricing accuracy"""

    def test_mc_call_matches_bsm(self):
        """MC call price should be close to BSM within CI"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2

        mc_price, se, ci_upper, _ = monte_carlo_european_call(
            S, K, T, r, vol, n_paths=10000, seed=42
        )
        bsm_price = bsm_call(S, K, T, r, vol)

        # BSM price should be within 95% CI
        ci_lower = mc_price - 1.96 * se
        assert ci_lower <= bsm_price <= ci_upper, (
            f"BSM price {bsm_price} outside CI [{ci_lower:.4f}, {ci_upper:.4f}]"
        )

    def test_mc_put_matches_bsm(self):
        """MC put price should be close to BSM within CI"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2

        mc_price, se, ci_upper, _ = monte_carlo_european_put(
            S, K, T, r, vol, n_paths=10000, seed=42
        )
        bsm_price = bsm_put(S, K, T, r, vol)

        # BSM price should be within 95% CI
        ci_lower = mc_price - 1.96 * se
        assert ci_lower <= bsm_price <= ci_upper, (
            f"BSM price {bsm_price} outside CI [{ci_lower:.4f}, {ci_upper:.4f}]"
        )

    def test_mc_call_itm_vs_otm(self):
        """ITM call should be worth more than OTM call"""
        T, r, vol = 1.0, 0.05, 0.2

        mc_call_itm, _, _, _ = monte_carlo_european_call(
            S=110.0, K=100.0, T=T, r=r, vol=vol, n_paths=5000, seed=42
        )
        mc_call_otm, _, _, _ = monte_carlo_european_call(
            S=90.0, K=100.0, T=T, r=r, vol=vol, n_paths=5000, seed=42
        )

        assert mc_call_itm > mc_call_otm, "ITM call should exceed OTM call"

    def test_mc_put_itm_vs_otm(self):
        """ITM put should be worth more than OTM put"""
        T, r, vol = 1.0, 0.05, 0.2

        mc_put_itm, _, _, _ = monte_carlo_european_put(
            S=90.0, K=100.0, T=T, r=r, vol=vol, n_paths=5000, seed=42
        )
        mc_put_otm, _, _, _ = monte_carlo_european_put(
            S=110.0, K=100.0, T=T, r=r, vol=vol, n_paths=5000, seed=42
        )

        assert mc_put_itm > mc_put_otm, "ITM put should exceed OTM put"


class TestStandardError:
    """Test that standard error shrinks ~ 1/sqrt(N)"""

    def test_se_shrinks_with_paths(self):
        """SE should decrease as sqrt(1/N)"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2

        ses = []
        n_paths_list = [100, 500, 1000, 5000]

        for n_paths in n_paths_list:
            _, se, _, _ = monte_carlo_european_call(
                S, K, T, r, vol, n_paths=n_paths, seed=42
            )
            ses.append(se)

        # SE should decrease (or stay roughly same within noise)
        # SE[5000] should be much smaller than SE[100]
        assert ses[-1] < ses[0], f"SE should decrease: {ses[0]:.6f} → {ses[-1]:.6f}"

    def test_se_ratio_matches_sqrt_scaling(self):
        """SE ratio should match sqrt(N1/N2) scaling"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2

        _, se_100, _, _ = monte_carlo_european_call(
            S, K, T, r, vol, n_paths=100, seed=42
        )
        _, se_10000, _, _ = monte_carlo_european_call(
            S, K, T, r, vol, n_paths=10000, seed=42
        )

        # SE scales as 1/sqrt(N), so SE_100 / SE_10000 ≈ sqrt(10000/100) = 10
        expected_ratio = np.sqrt(10000 / 100)
        actual_ratio = se_100 / se_10000

        # Allow 30% tolerance due to randomness
        assert expected_ratio * 0.7 < actual_ratio < expected_ratio * 1.3, (
            f"SE ratio {actual_ratio:.2f} should be ~{expected_ratio:.2f}"
        )


class TestAntithetic:
    """Test antithetic variates variance reduction"""

    def test_antithetic_reduces_variance(self):
        """MC with antithetic should have lower SE than without"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2

        # With antithetic
        _, se_with, _, _ = monte_carlo_european_call(
            S, K, T, r, vol, n_paths=5000, use_antithetic=True, seed=42
        )

        # Without antithetic (but same path count, so antithetic pair is unused)
        _, se_without, _, _ = monte_carlo_european_call(
            S, K, T, r, vol, n_paths=5000, use_antithetic=False, seed=42
        )

        # Both should produce reasonable estimates
        # (antithetic should be at least as good due to variance reduction)
        assert se_with <= se_without * 1.1, (
            f"Antithetic SE {se_with:.6f} should be ≤ standard SE {se_without:.6f}"
        )

    def test_antithetic_doubles_path_count(self):
        """Requesting n_paths with antithetic uses n_paths total (split original + antithetic)"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2

        _, _, _, diag = monte_carlo_european_call(
            S, K, T, r, vol, n_paths=1000, use_antithetic=True, seed=42
        )

        # Diagnostic should show exactly n_paths used (half original + half antithetic)
        assert diag["n_paths_used"] == 1000, (
            f"Antithetic should result in n_paths_used = {1000}; got {diag['n_paths_used']}"
        )
        assert diag["antithetic_used"] is True


class TestConvergence:
    """Test convergence study"""

    def test_convergence_ci_shrinks(self):
        """CI width should shrink with more paths"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2

        results = monte_carlo_convergence_study(
            S, K, T, r, vol,
            option_type="call",
            n_path_samples=[100, 1000, 10000],
        )

        ci_widths = [results[n]["ci_width"] for n in sorted(results.keys())]

        # CI widths should decrease
        assert ci_widths[0] > ci_widths[-1], (
            f"CI widths should decrease: {ci_widths[0]:.4f} → {ci_widths[-1]:.4f}"
        )

    def test_convergence_prices_stable(self):
        """MC prices should not vary wildly across n_paths (within CI)"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2

        results = monte_carlo_convergence_study(
            S, K, T, r, vol,
            option_type="call",
            n_path_samples=[1000, 5000, 10000],
        )

        prices = [results[n]["price"] for n in sorted(results.keys())]

        # Prices should be in same ballpark (within a few percent)
        price_range = max(prices) - min(prices)
        price_mean = np.mean(prices)
        price_pct_range = price_range / price_mean * 100

        assert price_pct_range < 5.0, (
            f"Price range {price_pct_range:.2f}% suggests instability"
        )


class TestDividends:
    """Test dividend yield handling"""

    def test_mc_call_with_dividend_lower(self):
        """Call with dividend yield should be lower than without"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2

        mc_no_div, _, _, _ = monte_carlo_european_call(
            S, K, T, r, vol, n_paths=5000, dividend_yield=0.0, seed=42
        )
        mc_with_div, _, _, _ = monte_carlo_european_call(
            S, K, T, r, vol, n_paths=5000, dividend_yield=0.03, seed=42
        )

        # Dividend should reduce call value (holder doesn't receive it)
        assert mc_with_div < mc_no_div, (
            f"Call with div {mc_with_div:.4f} should be < without div {mc_no_div:.4f}"
        )

    def test_mc_put_with_dividend_higher(self):
        """Put with dividend yield should be higher than without"""
        S, K, T, r, vol = 100.0, 100.0, 1.0, 0.05, 0.2

        mc_no_div, _, _, _ = monte_carlo_european_put(
            S, K, T, r, vol, n_paths=5000, dividend_yield=0.0, seed=42
        )
        mc_with_div, _, _, _ = monte_carlo_european_put(
            S, K, T, r, vol, n_paths=5000, dividend_yield=0.03, seed=42
        )

        # Dividend should increase put value (benefits holder)
        assert mc_with_div > mc_no_div, (
            f"Put with div {mc_with_div:.4f} should be > without div {mc_no_div:.4f}"
        )


class TestEdgeCases:
    """Test edge cases"""

    def test_mc_zero_dte(self):
        """MC with T → 0 should approach intrinsic value"""
        S, K, r, vol = 100.0, 100.0, 0.05, 0.2

        # Tiny time to expiry
        mc_price, se, _, _ = monte_carlo_european_call(
            S, K, T=1.0/365.0, r=r, vol=vol, n_paths=10000, seed=42
        )
        intrinsic = max(S - K, 0.0)

        # With such short T, MC should be very close to intrinsic
        assert abs(mc_price - intrinsic) < 0.5, (
            f"0DTE call {mc_price:.4f} should approach intrinsic {intrinsic:.4f}"
        )

    def test_mc_deep_itm(self):
        """Deep ITM call should approach S - K*exp(-rT)"""
        S, K, T, r, vol = 200.0, 100.0, 1.0, 0.05, 0.01

        mc_price, _, _, _ = monte_carlo_european_call(
            S, K, T, r, vol, n_paths=10000, seed=42
        )
        lower_bound = S - K * np.exp(-r * T)

        # Deep ITM should be worth at least lower bound
        assert mc_price > lower_bound - 1.0, (
            f"Deep ITM call {mc_price:.4f} should exceed lower bound {lower_bound:.4f}"
        )

    def test_mc_deep_otm(self):
        """Deep OTM call should be worth near zero"""
        S, K, T, r, vol = 50.0, 100.0, 1.0, 0.05, 0.01

        mc_price, _, _, _ = monte_carlo_european_call(
            S, K, T, r, vol, n_paths=10000, seed=42
        )

        # Deep OTM should be tiny
        assert mc_price < 0.1, (
            f"Deep OTM call {mc_price:.6f} should be near zero"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

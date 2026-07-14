"""
Cox-Ross-Rubinstein (CRR) binomial tree for American option pricing.

Model:
- Recombining tree: S_up = S * exp(vol * sqrt(dt)), S_down = 1/S_up
- Risk-neutral prob: p = (exp(r*dt) - d) / (u - d) where u = S_up/S, d = S_down/S
- Early exercise: At each node, V = max(intrinsic, continuation)
- Discrete dividends: Apply lump-sum cash flow at ex-dividend nodes

Stability:
- Time-steps chosen to avoid oscillation: N ≥ sqrt(3*vol²*T) / sqrt(vol²) ≈ sqrt(3*T)
- Risk-neutral probability bounded: 0 < p < 1 (satisfied for reasonable parameters)
- Use high-precision arithmetic throughout

Returns:
- Option price (scalar)
- Early-exercise indicator (boolean)
- Exercise boundary (optional: strikes where early exercise occurs)
"""

import numpy as np
from typing import Optional, Tuple, Dict
from options_engine.models.black_scholes import bsm_call, bsm_put


def binomial_american_call(
    S: float,
    K: float,
    T: float,
    r: float,
    vol: float,
    n_steps: Optional[int] = None,
    dividends: Optional[Dict[float, float]] = None,
    dividend_yield: float = 0.0,
) -> Tuple[float, bool]:
    """
    Price American call using binomial tree.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry.
        r: Risk-free rate.
        vol: Volatility.
        n_steps: Number of tree steps. If None, auto-computed as max(30, sqrt(T)*100).
        dividends: Dict mapping ex-dividend time (as fraction of T) to dividend amount.
                  E.g., {0.5: 2.0} = $2 dividend at T/2.
        dividend_yield: Continuous dividend yield (alternative to discrete dividends).

    Returns:
        (price, early_exercised) tuple.
        early_exercised: True if early exercise is optimal at root.
    """
    if n_steps is None:
        n_steps = max(30, int(np.ceil(np.sqrt(T) * 100)))

    dt = T / n_steps
    df = np.exp(-r * dt)
    growth = np.exp(vol * np.sqrt(dt))
    up_factor = growth
    down_factor = 1.0 / growth

    # Risk-neutral probability
    forward_growth = np.exp((r - dividend_yield) * dt)
    p = (forward_growth - down_factor) / (up_factor - down_factor)

    # Validate probability
    if not (0 < p < 1):
        raise ValueError(
            f"Risk-neutral probability p={p} out of bounds. "
            f"Check parameters: r={r}, vol={vol}, q={dividend_yield}"
        )

    # Build recombining tree
    # V[i][j] = option value at step i, up j times (down i-j times)
    V = np.zeros((n_steps + 1, n_steps + 1))
    S_tree = np.zeros((n_steps + 1, n_steps + 1))

    # Initialize terminal values (at expiry)
    for j in range(n_steps + 1):
        S_tree[n_steps, j] = S * (up_factor ** (n_steps - j)) * (down_factor ** j)
        # Apply any final dividend (if ex-date is exactly at T)
        S_node = S_tree[n_steps, j]
        V[n_steps, j] = max(S_node - K, 0.0)

    # Backward induction: work from expiry to present
    for i in range(n_steps - 1, -1, -1):
        for j in range(i + 1):
            # Spot price at this node: i steps, up j times, down i-j times
            S_node = S * (up_factor ** (i - j)) * (down_factor ** j)
            S_tree[i, j] = S_node

            # Check for dividend at this node
            current_time = i * dt
            S_after_div = S_node
            if dividends and current_time in dividends:
                S_after_div = S_node - dividends[current_time]
                S_after_div = max(S_after_div, 0.0)  # Ensure S stays positive

            # Continuation value: discounted expected future value
            V_up = V[i + 1, j]
            V_down = V[i + 1, j + 1]
            cont = df * (p * V_up + (1.0 - p) * V_down)

            # Intrinsic value
            intrinsic = max(S_after_div - K, 0.0)

            # Early exercise decision
            V[i, j] = max(intrinsic, cont)

    # Check if early exercise is optimal at root
    european_call = bsm_call(S, K, T, r, vol, dividend_yield=dividend_yield)
    early_exercised = V[0, 0] > european_call + 1e-6

    return V[0, 0], early_exercised


def binomial_american_put(
    S: float,
    K: float,
    T: float,
    r: float,
    vol: float,
    n_steps: Optional[int] = None,
    dividends: Optional[Dict[float, float]] = None,
    dividend_yield: float = 0.0,
) -> Tuple[float, bool]:
    """
    Price American put using binomial tree.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry.
        r: Risk-free rate.
        vol: Volatility.
        n_steps: Number of tree steps. If None, auto-computed.
        dividends: Dict mapping ex-dividend time to dividend amount.
        dividend_yield: Continuous dividend yield.

    Returns:
        (price, early_exercised) tuple.
    """
    if n_steps is None:
        n_steps = max(30, int(np.ceil(np.sqrt(T) * 100)))

    dt = T / n_steps
    df = np.exp(-r * dt)
    growth = np.exp(vol * np.sqrt(dt))
    up_factor = growth
    down_factor = 1.0 / growth

    # Risk-neutral probability
    forward_growth = np.exp((r - dividend_yield) * dt)
    p = (forward_growth - down_factor) / (up_factor - down_factor)

    if not (0 < p < 1):
        raise ValueError(f"Risk-neutral probability p={p} out of bounds")

    # Build tree
    V = np.zeros((n_steps + 1, n_steps + 1))

    # Terminal values at expiry
    for j in range(n_steps + 1):
        S_terminal = S * (up_factor ** (n_steps - j)) * (down_factor ** j)
        V[n_steps, j] = max(K - S_terminal, 0.0)

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        for j in range(i + 1):
            S_node = S * (up_factor ** (i - j)) * (down_factor ** j)

            # Apply dividend if present
            current_time = i * dt
            S_after_div = S_node
            if dividends and current_time in dividends:
                S_after_div = S_node - dividends[current_time]
                S_after_div = max(S_after_div, 0.0)

            # Continuation value
            V_up = V[i + 1, j]
            V_down = V[i + 1, j + 1]
            cont = df * (p * V_up + (1.0 - p) * V_down)

            # Intrinsic value
            intrinsic = max(K - S_after_div, 0.0)

            # Early exercise decision
            V[i, j] = max(intrinsic, cont)

    # Check if early exercise is optimal at root
    european_put = bsm_put(S, K, T, r, vol, dividend_yield=dividend_yield)
    early_exercised = V[0, 0] > european_put + 1e-6

    return V[0, 0], early_exercised


def binomial_tree_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    vol: float,
    option_type: str = "call",
    n_steps: Optional[int] = None,
    dividends: Optional[Dict[float, float]] = None,
    dividend_yield: float = 0.0,
    bump_size: float = 0.01,
) -> Dict[str, float]:
    """
    Compute Greeks for American options using finite differences on binomial tree.

    Note: These are numerical Greeks, not analytic. Bump size affects accuracy.

    Args:
        S, K, T, r, vol: Option parameters.
        option_type: "call" or "put".
        n_steps: Tree steps.
        dividends: Discrete dividend dict.
        dividend_yield: Continuous yield.
        bump_size: Perturbation size for finite differences.

    Returns:
        Dict with keys: delta, gamma, vega, theta, rho
    """
    # Base price
    if option_type.lower() == "call":
        price_0, _ = binomial_american_call(
            S, K, T, r, vol, n_steps, dividends, dividend_yield
        )
    else:
        price_0, _ = binomial_american_put(
            S, K, T, r, vol, n_steps, dividends, dividend_yield
        )

    greeks = {}

    # Delta: dPrice/dS (central difference)
    S_bump = S * bump_size
    if option_type.lower() == "call":
        price_up, _ = binomial_american_call(
            S + S_bump, K, T, r, vol, n_steps, dividends, dividend_yield
        )
        price_down, _ = binomial_american_call(
            S - S_bump, K, T, r, vol, n_steps, dividends, dividend_yield
        )
    else:
        price_up, _ = binomial_american_put(
            S + S_bump, K, T, r, vol, n_steps, dividends, dividend_yield
        )
        price_down, _ = binomial_american_put(
            S - S_bump, K, T, r, vol, n_steps, dividends, dividend_yield
        )

    greeks["delta"] = (price_up - price_down) / (2 * S_bump)

    # Gamma: d²Price/dS² (finite difference of delta)
    greeks["gamma"] = (price_up - 2 * price_0 + price_down) / (S_bump**2)

    # Vega: dPrice/dVol (bump vol by 0.01 = 1%)
    vol_bump = 0.01
    if option_type.lower() == "call":
        price_vol_up, _ = binomial_american_call(
            S, K, T, r, vol + vol_bump, n_steps, dividends, dividend_yield
        )
        price_vol_down, _ = binomial_american_call(
            S, K, T, r, vol - vol_bump, n_steps, dividends, dividend_yield
        )
    else:
        price_vol_up, _ = binomial_american_put(
            S, K, T, r, vol + vol_bump, n_steps, dividends, dividend_yield
        )
        price_vol_down, _ = binomial_american_put(
            S, K, T, r, vol - vol_bump, n_steps, dividends, dividend_yield
        )

    greeks["vega"] = (price_vol_up - price_vol_down) / (2 * vol_bump)

    # Theta: dPrice/dT (bump T by 1 day = 1/365)
    T_bump = 1.0 / 365.0
    if option_type.lower() == "call":
        price_t_up, _ = binomial_american_call(
            S, K, T + T_bump, r, vol, n_steps, dividends, dividend_yield
        )
    else:
        price_t_up, _ = binomial_american_put(
            S, K, T + T_bump, r, vol, n_steps, dividends, dividend_yield
        )

    # Theta is typically quoted as dPrice/dT (negative for longs)
    # We compute price change per day
    greeks["theta"] = (price_t_up - price_0) / T_bump

    # Rho: dPrice/dr (bump r by 1%)
    r_bump = 0.01
    if option_type.lower() == "call":
        price_r_up, _ = binomial_american_call(
            S, K, T, r + r_bump, vol, n_steps, dividends, dividend_yield
        )
        price_r_down, _ = binomial_american_call(
            S, K, T, r - r_bump, vol, n_steps, dividends, dividend_yield
        )
    else:
        price_r_up, _ = binomial_american_put(
            S, K, T, r + r_bump, vol, n_steps, dividends, dividend_yield
        )
        price_r_down, _ = binomial_american_put(
            S, K, T, r - r_bump, vol, n_steps, dividends, dividend_yield
        )

    greeks["rho"] = (price_r_up - price_r_down) / (2 * r_bump)

    return greeks

"""
Interactive Options Pricing Dashboard
Built with Streamlit + Options Engine
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from options_engine.models.black_scholes import bsm_call, bsm_put
from options_engine.greeks.analytic import delta, gamma, vega, theta, rho
from options_engine.chain.pricer import price_equity_option_chain, price_index_option_chain
from options_engine.solvers.implied_vol import implied_vol_call, implied_vol_put

st.set_page_config(page_title="Options Pricing Engine", layout="wide", initial_sidebar_state="expanded")

st.title("⚡ Options Pricing Engine")
st.markdown("**Professional options pricing for US equities and indices**")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    option_type = st.radio("Option Type", ["Call", "Put"])
    S = st.number_input("Spot Price (S)", value=100.0, min_value=0.01, step=1.0)
    K = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, step=1.0)
    T = st.number_input("Time to Expiry (years)", value=1.0, min_value=0.001, step=0.1)
    r = st.number_input("Risk-free Rate (r)", value=0.05, min_value=-0.1, max_value=0.5, step=0.01)
    vol = st.number_input("Volatility (σ)", value=0.2, min_value=0.001, max_value=2.0, step=0.05)
    q = st.number_input("Dividend Yield (q)", value=0.0, min_value=0.0, max_value=0.2, step=0.01)

    st.divider()
    calc_type = st.radio("Calculate", ["Price & Greeks", "Implied Vol", "Option Chain", "Greeks Profile"])

# Main content area
if calc_type == "Price & Greeks":
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Option Price")
        if option_type == "Call":
            price = bsm_call(S, K, T, r, vol, dividend_yield=q)
            st.metric("Call Price", f"${price:.2f}")
            d = delta(S, K, T, r, vol, option_type="call", dividend_yield=q)
        else:
            price = bsm_put(S, K, T, r, vol, dividend_yield=q)
            st.metric("Put Price", f"${price:.2f}")
            d = delta(S, K, T, r, vol, option_type="put", dividend_yield=q)

        st.metric("Moneyness", f"{S/K:.2f}x", delta="ITM" if (option_type == "Call" and S > K) or (option_type == "Put" and S < K) else "OTM")

    with col2:
        st.subheader("📈 Greeks (Sensitivity)")
        g = gamma(S, K, T, r, vol, dividend_yield=q)
        v = vega(S, K, T, r, vol, dividend_yield=q)
        th = theta(S, K, T, r, vol, option_type=option_type.lower(), dividend_yield=q)
        rh = rho(S, K, T, r, vol, option_type=option_type.lower(), dividend_yield=q)

        st.metric("Delta (Δ)", f"{d:.4f}", "directional")
        st.metric("Gamma (Γ)", f"{g:.6f}", "acceleration")
        st.metric("Vega (ν)", f"{v:.4f}", "vol sensitivity")
        st.metric("Theta (Θ)", f"{th:.4f}", "time decay")
        st.metric("Rho (ρ)", f"{rh:.4f}", "rate sensitivity")

    # Greeks dashboard
    st.subheader("📉 Greeks Across Strikes")
    strikes_range = np.linspace(S * 0.8, S * 1.2, 50)
    greeks_data = {
        "Strike": strikes_range,
        "Delta": [delta(S, k, T, r, vol, option_type=option_type.lower(), dividend_yield=q) for k in strikes_range],
        "Gamma": [gamma(S, k, T, r, vol, dividend_yield=q) for k in strikes_range],
        "Vega": [vega(S, k, T, r, vol, dividend_yield=q) for k in strikes_range],
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, greek in zip(axes, ["Delta", "Gamma", "Vega"]):
        ax.plot(greeks_data["Strike"], greeks_data[greek], linewidth=2, color="steelblue")
        ax.axvline(S, color="red", linestyle="--", alpha=0.5, label=f"Spot (${S:.0f})")
        ax.axvline(K, color="green", linestyle="--", alpha=0.5, label=f"Strike (${K:.0f})")
        ax.set_xlabel("Strike")
        ax.set_ylabel(greek)
        ax.set_title(f"{greek} Profile")
        ax.grid(True, alpha=0.3)
        ax.legend()

    st.pyplot(fig)

elif calc_type == "Implied Vol":
    st.subheader("🔍 Implied Volatility Solver")

    market_price = st.number_input("Market Option Price", value=10.0, min_value=0.0, step=0.1)

    if st.button("Solve for IV"):
        if option_type == "Call":
            iv = implied_vol_call(market_price, S, K, T, r, q=q)
        else:
            iv = implied_vol_put(market_price, S, K, T, r, q=q)

        if iv is not None:
            st.success(f"✅ **Implied Volatility: {iv:.2%}**")

            # Round-trip validation
            recovered_price = bsm_call(S, K, T, r, iv, dividend_yield=q) if option_type == "Call" else bsm_put(S, K, T, r, iv, dividend_yield=q)
            st.info(f"🔄 Round-trip: ${market_price:.2f} → IV={iv:.2%} → ${recovered_price:.2f}")
        else:
            st.error("❌ No solution found (possible arbitrage violation)")

elif calc_type == "Option Chain":
    st.subheader("📊 Option Chain Pricer")

    col1, col2 = st.columns(2)
    with col1:
        n_strikes = st.slider("Number of Strikes", 5, 50, 11)
        n_expiries = st.slider("Number of Expiries", 2, 12, 4)

    with col2:
        chain_type = st.radio("Chain Type", ["Equity", "Index"])

    # Generate strikes and expiries
    strikes = np.linspace(S * 0.85, S * 1.15, n_strikes)
    expiries = np.linspace(0.1, 2.0, n_expiries)

    if st.button("Price Chain"):
        with st.spinner("Pricing chain..."):
            if chain_type == "Equity":
                chain = price_equity_option_chain(S, strikes, expiries, r, q=q, option_type=option_type.lower())
            else:
                chain = price_index_option_chain(S, strikes, expiries, r, q=q, option_type=option_type.lower())

        st.success(f"✅ Priced {len(chain)} options in milliseconds")

        # Pivot for better display
        chain_pivot = chain.pivot_table(
            index="strike",
            columns="expiry",
            values="price",
            aggfunc="first"
        )

        st.dataframe(chain_pivot.style.format("${:.2f}"), use_container_width=True)

        # Heatmap
        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(chain_pivot.values, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(chain_pivot.columns)))
        ax.set_yticks(range(len(chain_pivot.index)))
        ax.set_xticklabels([f"{t:.2f}y" for t in chain_pivot.columns])
        ax.set_yticklabels([f"${s:.0f}" for s in chain_pivot.index])
        ax.set_xlabel("Expiry")
        ax.set_ylabel("Strike")
        ax.set_title(f"{option_type} Prices Heatmap")
        plt.colorbar(im, ax=ax, label="Price ($)")
        st.pyplot(fig)

elif calc_type == "Greeks Profile":
    st.subheader("📊 Greeks Profile Across Time & Spot")

    # Time series
    st.markdown("**Greeks Over Time (Spot Constant)**")
    times = np.linspace(0.01, T, 50)

    greeks_time = {
        "Time": times,
        "Delta": [delta(S, K, t, r, vol, option_type=option_type.lower(), dividend_yield=q) for t in times],
        "Gamma": [gamma(S, K, t, r, vol, dividend_yield=q) for t in times],
        "Theta": [theta(S, K, t, r, vol, option_type=option_type.lower(), dividend_yield=q) for t in times],
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, greek in zip(axes, ["Delta", "Gamma", "Theta"]):
        ax.plot(greeks_time["Time"], greeks_time[greek], linewidth=2, color="steelblue")
        ax.set_xlabel("Time to Expiry (years)")
        ax.set_ylabel(greek)
        ax.set_title(f"{greek} vs Time")
        ax.grid(True, alpha=0.3)

    st.pyplot(fig)

    # Spot series
    st.markdown("**Greeks Across Spot Prices (Time Constant)**")
    spots = np.linspace(S * 0.7, S * 1.3, 50)

    greeks_spot = {
        "Spot": spots,
        "Delta": [delta(s, K, T, r, vol, option_type=option_type.lower(), dividend_yield=q) for s in spots],
        "Gamma": [gamma(s, K, T, r, vol, dividend_yield=q) for s in spots],
        "Vega": [vega(s, K, T, r, vol, dividend_yield=q) for s in spots],
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, greek in zip(axes, ["Delta", "Gamma", "Vega"]):
        ax.plot(greeks_spot["Spot"], greeks_spot[greek], linewidth=2, color="darkgreen")
        ax.axvline(S, color="red", linestyle="--", alpha=0.5, label="Current Spot")
        ax.axvline(K, color="orange", linestyle="--", alpha=0.5, label="Strike")
        ax.set_xlabel("Spot Price ($)")
        ax.set_ylabel(greek)
        ax.set_title(f"{greek} vs Spot")
        ax.grid(True, alpha=0.3)
        ax.legend()

    st.pyplot(fig)

# Footer
st.divider()
st.markdown("""
---
**Options Engine** | [GitHub](https://github.com/NyunnSintHtoo/options-engine) | All 9 Stages Complete ✅
- Stage 1: Black-Scholes-Merton + Greeks
- Stage 2: American Options + Dividends
- Stage 3: Monte Carlo Pricing
- Stage 4: Implied-Vol Solver
- Stage 5: Vectorized Chain Pricer
- Stage 6-9: Real-Time, R Implementation, Optimization, Professional Repo
""")

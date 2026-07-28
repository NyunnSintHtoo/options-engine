"""
Options Analytics Engine — Interactive Dashboard

Black-Scholes, Monte Carlo, and binomial pricing with full Greeks,
implied-vol solving, and vectorized chain pricing.
"""

import sys
from pathlib import Path

# Make the engine importable straight from the repo (no pip install needed,
# e.g. on Streamlit Community Cloud).
sys.path.insert(0, str(Path(__file__).resolve().parent / "python" / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from options_engine.chain.pricer import price_equity_option_chain, price_index_option_chain
from options_engine.greeks.analytic import delta, gamma, rho, theta, vega
from options_engine.models.binomial import binomial_american_call, binomial_american_put
from options_engine.models.black_scholes import bsm_call, bsm_put
from options_engine.models.monte_carlo import monte_carlo_european_call, monte_carlo_european_put
from options_engine.solvers.implied_vol import implied_vol_call, implied_vol_put

# ---------------------------------------------------------------------------
# Design tokens (dark surface palette)
# ---------------------------------------------------------------------------
SURFACE = "#1a1a19"
PAGE = "#0d0d0d"
INK = "#ffffff"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
BASELINE = "#383835"
BLUE = "#3987e5"
ORANGE = "#d95926"
AQUA = "#199e70"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

st.set_page_config(
    page_title="Options Analytics Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp, [data-testid="stSidebar"] {
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1250px; }

/* Hero */
.hero-badge {
    display: inline-block; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase; color: #5598e7;
    background: rgba(57, 135, 229, 0.12); border: 1px solid rgba(57, 135, 229, 0.35);
    padding: 0.25rem 0.7rem; border-radius: 999px; margin-bottom: 0.6rem;
}
.hero-title { font-size: 2.1rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em; margin: 0; }
.hero-sub { color: #898781; font-size: 0.95rem; margin-top: 0.35rem; }

/* Stat cards */
.stat-card {
    background: #1a1a19; border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px; padding: 1rem 1.15rem; height: 100%;
}
.stat-label { color: #898781; font-size: 0.74rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; }
.stat-value { color: #ffffff; font-size: 1.55rem; font-weight: 700; letter-spacing: -0.01em; margin-top: 0.15rem; }
.stat-note { color: #c3c2b7; font-size: 0.78rem; margin-top: 0.2rem; }
.stat-accent { color: #3987e5; }
.stat-good { color: #0ca30c; }
.stat-warn { color: #d03b3b; }

.price-card {
    background: linear-gradient(135deg, rgba(57,135,229,0.16) 0%, #1a1a19 55%);
    border: 1px solid rgba(57,135,229,0.35); border-radius: 16px;
    padding: 1.3rem 1.4rem;
}
.price-card .stat-value { font-size: 2.3rem; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 0.25rem; border-bottom: 1px solid rgba(255,255,255,0.10); }
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 8px 8px 0 0;
    color: #898781; font-weight: 500; padding: 0.55rem 1rem;
}
.stTabs [aria-selected="true"] { color: #ffffff !important; }

/* Sidebar */
[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.08); }
[data-testid="stSidebar"] .stMarkdown h3 { font-size: 0.8rem; letter-spacing: 0.07em; text-transform: uppercase; color: #898781; }

/* Tables */
[data-testid="stDataFrame"] { border: 1px solid rgba(255,255,255,0.10); border-radius: 12px; }
hr { border-color: rgba(255,255,255,0.08); }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------
def base_layout(fig: go.Figure, *, height: int = 360, ylab: str = "", xlab: str = "") -> go.Figure:
    fig.update_layout(
        template=None,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        height=height,
        margin=dict(l=54, r=24, t=36, b=44),
        font=dict(family="Inter, system-ui, sans-serif", color=INK_2, size=12),
        hoverlabel=dict(bgcolor=PAGE, bordercolor=BASELINE, font=dict(color=INK, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    fig.update_xaxes(
        title=dict(text=xlab, font=dict(color=MUTED, size=11)),
        gridcolor=GRID, zerolinecolor=BASELINE, linecolor=BASELINE,
        tickfont=dict(color=MUTED, size=11), showspikes=True,
        spikecolor=BASELINE, spikethickness=1, spikedash="dot",
    )
    fig.update_yaxes(
        title=dict(text=ylab, font=dict(color=MUTED, size=11)),
        gridcolor=GRID, zerolinecolor=BASELINE, linecolor=BASELINE,
        tickfont=dict(color=MUTED, size=11),
    )
    return fig


def stat_card(label: str, value: str, note: str = "", tone: str = "") -> str:
    note_html = f'<div class="stat-note">{note}</div>' if note else ""
    return (
        f'<div class="stat-card"><div class="stat-label">{label}</div>'
        f'<div class="stat-value {tone}">{value}</div>{note_html}</div>'
    )


# ---------------------------------------------------------------------------
# Sidebar — market parameters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Contract")
    option_type = st.segmented_control("Option type", ["Call", "Put"], default="Call")
    if option_type is None:
        option_type = "Call"
    is_call = option_type == "Call"

    st.markdown("### Market parameters")
    S = st.number_input("Spot price S ($)", value=100.0, min_value=0.01, step=1.0)
    K = st.number_input("Strike price K ($)", value=100.0, min_value=0.01, step=1.0)
    T = st.slider("Time to expiry T (years)", 0.02, 3.0, 1.0, 0.02)
    vol = st.slider("Volatility σ", 0.05, 1.0, 0.20, 0.01, format="%.2f")
    r = st.slider("Risk-free rate r", -0.02, 0.15, 0.05, 0.005, format="%.3f")
    q = st.slider("Dividend yield q", 0.0, 0.10, 0.0, 0.005, format="%.3f")

    st.divider()
    st.caption("Vectorized NumPy engines · 97 CI tests · cross-validated against R")

opt = option_type.lower()

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    f"""
<div class="hero-badge">Black-Scholes · Monte Carlo · Binomial</div>
<h1 class="hero-title">Options Analytics Engine</h1>
<div class="hero-sub">Institutional-style pricing, Greeks, implied vol, and chain analytics —
priced live as you move the parameters.</div>
""",
    unsafe_allow_html=True,
)
st.markdown("")

# Core pricing (always computed — cheap)
bs_price = bsm_call(S, K, T, r, vol, dividend_yield=q) if is_call else bsm_put(S, K, T, r, vol, dividend_yield=q)
d_ = delta(S, K, T, r, vol, option_type=opt, dividend_yield=q)
g_ = gamma(S, K, T, r, vol, dividend_yield=q)
v_ = vega(S, K, T, r, vol, dividend_yield=q)
th_ = theta(S, K, T, r, vol, option_type=opt, dividend_yield=q)
rh_ = rho(S, K, T, r, vol, option_type=opt, dividend_yield=q)

intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
time_value = bs_price - intrinsic
itm = (is_call and S > K) or (not is_call and S < K)
breakeven = K + bs_price if is_call else K - bs_price

tab_price, tab_models, tab_iv, tab_chain, tab_greeks = st.tabs(
    ["  Pricing & Greeks  ", "  Model Lab  ", "  Implied Vol  ", "  Option Chain  ", "  Greeks Explorer  "]
)

# ===========================================================================
# TAB 1 — Pricing & Greeks
# ===========================================================================
with tab_price:
    c1, c2, c3, c4 = st.columns([1.35, 1, 1, 1])
    with c1:
        st.markdown(
            f'<div class="price-card"><div class="stat-label">Black-Scholes {option_type} price</div>'
            f'<div class="stat-value stat-accent">${bs_price:,.2f}</div>'
            f'<div class="stat-note">intrinsic ${intrinsic:,.2f} · time value ${time_value:,.2f}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            stat_card("Moneyness S / K", f"{S / K:.2f}×", "in the money" if itm else "out of the money",
                      "stat-good" if itm else ""),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(stat_card("Breakeven at expiry", f"${breakeven:,.2f}"), unsafe_allow_html=True)
    with c4:
        st.markdown(stat_card("Daily theta", f"${th_ / 365.0:,.4f}", "per calendar day"), unsafe_allow_html=True)

    st.markdown("")
    gc = st.columns(5)
    greek_cards = [
        ("Delta Δ", f"{d_:.4f}", "$ per $1 spot move"),
        ("Gamma Γ", f"{g_:.5f}", "Δ change per $1"),
        ("Vega ν", f"{v_:.4f}", "$ per 1.00 vol"),
        ("Theta Θ", f"{th_:.4f}", "$ per year"),
        ("Rho ρ", f"{rh_:.4f}", "$ per 1.00 rate"),
    ]
    for col, (lbl, val, note) in zip(gc, greek_cards):
        with col:
            st.markdown(stat_card(lbl, val, note), unsafe_allow_html=True)

    st.markdown("")
    left, right = st.columns(2)

    with left:
        st.markdown("##### Value today vs payoff at expiry")
        spots = np.linspace(max(0.4 * K, 1e-6), 1.6 * K, 160)
        value_today = np.array([
            bsm_call(s, K, T, r, vol, dividend_yield=q) if is_call else bsm_put(s, K, T, r, vol, dividend_yield=q)
            for s in spots
        ])
        payoff = np.maximum(spots - K, 0) if is_call else np.maximum(K - spots, 0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spots, y=payoff, name="Payoff at expiry",
                                 line=dict(color=MUTED, width=2, dash="dot"),
                                 hovertemplate="$%{y:.2f}<extra>payoff</extra>"))
        fig.add_trace(go.Scatter(x=spots, y=value_today, name="Value today",
                                 line=dict(color=BLUE, width=2.4),
                                 hovertemplate="$%{y:.2f}<extra>value today</extra>"))
        fig.add_vline(x=S, line=dict(color=BASELINE, width=1, dash="dash"))
        fig.add_annotation(x=S, y=float(value_today.max()), text="spot", showarrow=False,
                           font=dict(color=MUTED, size=11), yshift=10)
        base_layout(fig, xlab="Spot price ($)", ylab="Option value ($)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.markdown("##### Delta and gamma across strikes")
        strikes = np.linspace(0.8 * S, 1.2 * S, 90)
        deltas = [delta(S, k, T, r, vol, option_type=opt, dividend_yield=q) for k in strikes]
        gammas = [gamma(S, k, T, r, vol, dividend_yield=q) for k in strikes]
        gmax = max(gammas) or 1.0
        dmax = max(abs(min(deltas)), abs(max(deltas))) or 1.0
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=strikes, y=deltas, name="Delta",
                                 line=dict(color=BLUE, width=2.4),
                                 hovertemplate="%{y:.4f}<extra>delta</extra>"))
        fig.add_trace(go.Scatter(x=strikes, y=[g / gmax * dmax for g in gammas], name="Gamma (scaled)",
                                 line=dict(color=ORANGE, width=2.4),
                                 hovertemplate="scaled<extra>gamma</extra>"))
        fig.add_vline(x=K, line=dict(color=BASELINE, width=1, dash="dash"))
        fig.add_annotation(x=K, y=dmax, text="strike", showarrow=False,
                           font=dict(color=MUTED, size=11), yshift=8)
        base_layout(fig, xlab="Strike ($)", ylab="Delta (gamma rescaled)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ===========================================================================
# TAB 2 — Model Lab (BS vs Monte Carlo vs Binomial)
# ===========================================================================
with tab_models:
    st.markdown("##### Three engines, one contract")
    st.caption(
        "European closed-form Black-Scholes, Monte Carlo with antithetic + control-variate "
        "variance reduction, and a CRR binomial tree priced as an American option."
    )

    mc_col1, mc_col2, _ = st.columns([1, 1, 2])
    with mc_col1:
        n_paths = st.select_slider("MC paths", options=[5_000, 10_000, 25_000, 50_000, 100_000], value=25_000)
    with mc_col2:
        seed = st.number_input("MC seed", value=42, min_value=0, step=1)

    mc_fn = monte_carlo_european_call if is_call else monte_carlo_european_put
    mc_price, mc_se, mc_ci_hi, _diag = mc_fn(S, K, T, r, vol, n_paths=int(n_paths),
                                             dividend_yield=q, seed=int(seed))
    mc_ci_lo = mc_price - 1.96 * mc_se

    bin_fn = binomial_american_call if is_call else binomial_american_put
    bin_price, early_ex = bin_fn(S, K, T, r, vol, dividend_yield=q)
    early_premium = bin_price - bs_price

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(stat_card("Black-Scholes (European)", f"${bs_price:,.4f}", "closed form", "stat-accent"),
                    unsafe_allow_html=True)
    with r2:
        st.markdown(
            stat_card("Monte Carlo (European)", f"${mc_price:,.4f}",
                      f"SE {mc_se:.4f} · 95% CI [{mc_ci_lo:,.4f}, {mc_ci_hi:,.4f}]"),
            unsafe_allow_html=True,
        )
    with r3:
        st.markdown(
            stat_card("Binomial CRR (American)", f"${bin_price:,.4f}",
                      "early exercise optimal" if early_ex else "no early exercise",
                      "stat-good" if early_ex else ""),
            unsafe_allow_html=True,
        )
    with r4:
        st.markdown(
            stat_card("Early-exercise premium", f"${max(early_premium, 0):,.4f}",
                      "American − European value"),
            unsafe_allow_html=True,
        )

    st.markdown("")
    lab_l, lab_r = st.columns(2)

    with lab_l:
        st.markdown("##### Engine agreement")
        names = ["Black-Scholes", "Monte Carlo", "Binomial (Am.)"]
        vals = [bs_price, mc_price, bin_price]
        fig = go.Figure(
            go.Bar(
                x=names, y=vals,
                marker=dict(color=[BLUE, ORANGE, AQUA], cornerradius=4,
                            line=dict(color=SURFACE, width=2)),
                width=0.55,
                error_y=dict(type="data", array=[0, 1.96 * mc_se, 0],
                             color=INK_2, thickness=1.4, width=6),
                text=[f"${v:,.3f}" for v in vals], textposition="outside",
                textfont=dict(color=INK_2, size=12),
                hovertemplate="$%{y:.4f}<extra>%{x}</extra>",
            )
        )
        base_layout(fig, ylab="Option price ($)")
        fig.update_layout(hovermode="closest", showlegend=False)
        pad = max(vals) * 0.12
        fig.update_yaxes(range=[0, max(vals) + pad])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with lab_r:
        st.markdown("##### Monte Carlo convergence (SE ~ 1/√N)")
        with st.spinner("Running convergence study..."):
            path_grid = [1_000, 2_500, 5_000, 10_000, 25_000, 50_000]
            ses, prices = [], []
            for n in path_grid:
                p, se, _, _ = mc_fn(S, K, T, r, vol, n_paths=n, dividend_yield=q, seed=int(seed))
                prices.append(p)
                ses.append(se)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=path_grid, y=ses, name="Standard error",
            mode="lines+markers", line=dict(color=BLUE, width=2.4),
            marker=dict(size=8, color=BLUE, line=dict(color=SURFACE, width=2)),
            hovertemplate="SE %{y:.5f} at %{x:,} paths<extra></extra>",
        ))
        ref = ses[0] * np.sqrt(path_grid[0] / np.array(path_grid, dtype=float))
        fig.add_trace(go.Scatter(
            x=path_grid, y=ref, name="1/√N reference",
            line=dict(color=MUTED, width=1.6, dash="dot"), hoverinfo="skip",
        ))
        base_layout(fig, xlab="Number of paths", ylab="Standard error ($)")
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ===========================================================================
# TAB 3 — Implied Volatility
# ===========================================================================
with tab_iv:
    st.markdown("##### Implied volatility solver")
    st.caption("Invert Black-Scholes: given a market premium, recover the volatility the market is pricing in.")

    iv_l, iv_r = st.columns([1, 1.6])
    with iv_l:
        market_price = st.number_input("Market option price ($)", value=round(float(bs_price), 2),
                                       min_value=0.0, step=0.1)
        solve = st.button("Solve for IV", type="primary", use_container_width=True)
        if solve:
            iv = (implied_vol_call if is_call else implied_vol_put)(market_price, S, K, T, r, q=q)
            if iv is not None:
                recovered = (bsm_call if is_call else bsm_put)(S, K, T, r, iv, dividend_yield=q)
                st.session_state["iv_result"] = (market_price, iv, recovered)
            else:
                st.session_state["iv_result"] = (market_price, None, None)

        result = st.session_state.get("iv_result")
        if result:
            mp, iv, recovered = result
            if iv is None:
                st.markdown(
                    stat_card("No solution", "—", "price violates arbitrage bounds", "stat-warn"),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(stat_card("Implied volatility", f"{iv:.2%}", "annualized", "stat-accent"),
                            unsafe_allow_html=True)
                st.markdown("")
                st.markdown(
                    stat_card("Round-trip check", f"${recovered:,.4f}",
                              f"repriced from IV vs market ${mp:,.2f}", "stat-good"),
                    unsafe_allow_html=True,
                )

    with iv_r:
        st.markdown("##### Premium vs volatility (current contract)")
        vols = np.linspace(0.02, 1.0, 120)
        prices_v = [(bsm_call if is_call else bsm_put)(S, K, T, r, v0, dividend_yield=q) for v0 in vols]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=vols * 100, y=prices_v, name="BS premium",
            line=dict(color=BLUE, width=2.4),
            hovertemplate="$%{y:.2f} at %{x:.0f}% vol<extra></extra>",
        ))
        result = st.session_state.get("iv_result")
        if result and result[1] is not None:
            fig.add_trace(go.Scatter(
                x=[result[1] * 100], y=[result[0]], name="Solved IV",
                mode="markers",
                marker=dict(size=12, color=ORANGE, line=dict(color=SURFACE, width=2)),
                hovertemplate="IV %{x:.1f}%% → $%{y:.2f}<extra></extra>",
            ))
        base_layout(fig, xlab="Volatility (%)", ylab="Option premium ($)", height=380)
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ===========================================================================
# TAB 4 — Option Chain
# ===========================================================================
with tab_chain:
    st.markdown("##### Vectorized chain pricer")
    st.caption("Prices the full strike × expiry grid in a single vectorized pass — 7,000+ contracts/sec.")

    cc1, cc2, cc3 = st.columns([1, 1, 1])
    with cc1:
        n_strikes = st.slider("Strikes", 5, 41, 13, 2)
    with cc2:
        n_expiries = st.slider("Expiries", 2, 12, 6)
    with cc3:
        chain_type = st.selectbox("Underlying type", ["Equity", "Index"])

    strikes = np.linspace(S * 0.85, S * 1.15, n_strikes)
    expiries = np.linspace(0.1, 2.0, n_expiries)

    import time as _time

    chain_fn = price_equity_option_chain if chain_type == "Equity" else price_index_option_chain
    t0 = _time.perf_counter()
    chain = chain_fn(S, strikes, expiries, r, q=q, option_type=opt)
    elapsed_ms = (_time.perf_counter() - t0) * 1000

    st.markdown(
        stat_card(
            "Chain priced",
            f"{len(chain):,} contracts",
            f"{elapsed_ms:.1f} ms · {len(chain) / max(elapsed_ms, 1e-9) * 1000:,.0f} contracts/sec",
            "stat-accent",
        ),
        unsafe_allow_html=True,
    )
    st.markdown("")

    pivot = chain.pivot_table(index="strike", columns="expiry", values="price", aggfunc="first")
    pivot = pivot.sort_index(ascending=False)

    hm_l, hm_r = st.columns([1.35, 1])
    with hm_l:
        st.markdown("##### Price surface")
        fig = go.Figure(
            go.Heatmap(
                z=pivot.values,
                x=[f"{t0_:.2f}y" for t0_ in pivot.columns],
                y=[f"${s0:.0f}" for s0 in pivot.index],
                colorscale=[
                    [0.0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"],
                    [0.75, "#1c5cab"], [1.0, "#0d366b"],
                ],
                xgap=2, ygap=2,
                colorbar=dict(title=dict(text="$", font=dict(color=MUTED)),
                              tickfont=dict(color=MUTED), outlinewidth=0, thickness=12),
                hovertemplate="strike %{y} · expiry %{x}<br>$%{z:.2f}<extra></extra>",
            )
        )
        base_layout(fig, xlab="Time to expiry", ylab="Strike", height=430)
        fig.update_layout(hovermode="closest")
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with hm_r:
        st.markdown("##### Chain table")
        st.dataframe(
            pivot.style.format("${:.2f}").background_gradient(cmap="Blues", axis=None),
            use_container_width=True, height=430,
        )

# ===========================================================================
# TAB 5 — Greeks Explorer
# ===========================================================================
with tab_greeks:
    st.markdown("##### How the Greeks move")
    st.caption("Left: sensitivity across spot at the current expiry. Right: evolution as expiry approaches.")

    greek_choice = st.segmented_control(
        "Greek", ["Delta", "Gamma", "Vega", "Theta"], default="Delta"
    ) or "Delta"

    def greek_val(greek: str, s: float, k: float, t: float) -> float:
        if greek == "Delta":
            return delta(s, k, t, r, vol, option_type=opt, dividend_yield=q)
        if greek == "Gamma":
            return gamma(s, k, t, r, vol, dividend_yield=q)
        if greek == "Vega":
            return vega(s, k, t, r, vol, dividend_yield=q)
        return theta(s, k, t, r, vol, option_type=opt, dividend_yield=q)

    ge_l, ge_r = st.columns(2)

    with ge_l:
        spots = np.linspace(0.7 * S, 1.3 * S, 110)
        ys = [greek_val(greek_choice, s0, K, T) for s0 in spots]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=spots, y=ys, name=greek_choice,
            line=dict(color=BLUE, width=2.4),
            hovertemplate="%{y:.5f}<extra>" + greek_choice + "</extra>",
        ))
        fig.add_vline(x=S, line=dict(color=BASELINE, width=1, dash="dash"))
        fig.add_vline(x=K, line=dict(color=GRID, width=1, dash="dot"))
        fig.add_annotation(x=S, y=max(ys), text="spot", showarrow=False, font=dict(color=MUTED, size=11), yshift=8)
        fig.add_annotation(x=K, y=min(ys), text="strike", showarrow=False, font=dict(color=MUTED, size=11), yshift=-8)
        base_layout(fig, xlab="Spot price ($)", ylab=greek_choice, height=400)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with ge_r:
        times = np.linspace(0.01, max(T, 0.02), 110)
        ys_t = [greek_val(greek_choice, S, K, t0_) for t0_ in times]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times, y=ys_t, name=greek_choice,
            line=dict(color=ORANGE, width=2.4),
            hovertemplate="%{y:.5f}<extra>" + greek_choice + "</extra>",
        ))
        base_layout(fig, xlab="Time to expiry (years)", ylab=greek_choice, height=400)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("")
st.markdown(
    f"""
<div style="border-top: 1px solid rgba(255,255,255,0.08); padding-top: 1rem; margin-top: 1.5rem;
            display: flex; justify-content: space-between; color: #898781; font-size: 0.8rem;">
  <span><b style="color:#c3c2b7;">Options Analytics Engine</b> · Python (NumPy · SciPy) + R cross-validation</span>
  <span>Black-Scholes · Monte Carlo · Binomial · Implied Vol · Chain Pricer</span>
</div>
""",
    unsafe_allow_html=True,
)

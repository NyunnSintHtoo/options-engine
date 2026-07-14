"""Real-time market feed simulator with chain re-pricing."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Generator, Tuple
from options_engine.chain.pricer import OptionChainPricer


class MarketTickFeed:
    """Simulated market-data tick stream."""

    def __init__(
        self,
        S0: float,
        drift: float = 0.05,
        vol: float = 0.2,
        tick_interval_ms: int = 100,
        seed: int = 42,
    ):
        """
        Initialize market feed.

        Args:
            S0: Initial spot price.
            drift: Annual drift (mean return).
            vol: Annual volatility.
            tick_interval_ms: Milliseconds between ticks.
            seed: Random seed.
        """
        self.S0 = S0
        self.drift = drift
        self.vol = vol
        self.tick_interval_ms = tick_interval_ms
        np.random.seed(seed)

    def generate_ticks(
        self,
        n_ticks: int,
        market_hours_only: bool = True,
    ) -> Generator[Tuple[float, datetime], None, None]:
        """
        Generate market ticks with random walk.

        Args:
            n_ticks: Number of ticks to generate.
            market_hours_only: If True, skip overnight gap.

        Yields:
            (spot_price, timestamp) tuples.
        """
        S = self.S0
        dt = self.tick_interval_ms / 1000 / 252 / 6.5  # Convert to trading years
        drift_term = (self.drift - 0.5 * self.vol**2) * dt
        diffusion_term = self.vol * np.sqrt(dt)

        now = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)  # Market open

        for i in range(n_ticks):
            # Random return
            Z = np.random.randn()
            dS_S = drift_term + diffusion_term * Z
            S = S * np.exp(dS_S)

            yield S, now

            # Advance time
            now += timedelta(milliseconds=self.tick_interval_ms)


class ChainRepriceStream:
    """Re-price option chain on each market tick."""

    def __init__(
        self,
        feed: MarketTickFeed,
        strikes: np.ndarray,
        expiries: np.ndarray,
        r: float = 0.05,
        q: float = 0.0,
    ):
        """
        Initialize re-pricer.

        Args:
            feed: Market tick feed.
            strikes: Option chain strikes.
            expiries: Option chain expiries.
            r: Risk-free rate.
            q: Dividend yield.
        """
        self.feed = feed
        self.strikes = strikes
        self.expiries = expiries
        self.r = r
        self.q = q

    def stream_prices(
        self,
        n_ticks: int,
        option_type: str = "call",
    ) -> Generator[Tuple[float, pd.DataFrame, float], None, None]:
        """
        Stream re-priced chains.

        Yields:
            (spot, chain_df, latency_ms) tuples.
        """
        import time

        for S, ts in self.feed.generate_ticks(n_ticks):
            pricer = OptionChainPricer(S, self.r, self.q, option_type)

            start = time.perf_counter()
            chain = pricer.price_chain(self.strikes, self.expiries, compute_greeks=True)
            latency = (time.perf_counter() - start) * 1000

            yield S, chain, latency

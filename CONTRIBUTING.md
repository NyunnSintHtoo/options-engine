# Contributing to Options Engine

## Development Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/options-engine.git
cd options-engine

# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
python -m pytest python/tests/ -v

# Run benchmarks
python python/benchmarks/bench_single_option.py
```

## Project Structure

- **`python/src/options_engine/`**: Main engine
  - `models/`: Pricing models (BSM, binomial, MC)
  - `greeks/`: Greeks computation
  - `solvers/`: Implied vol, calibration
  - `utils/`: Numerics, validation
- **`python/tests/`**: pytest unit tests
- **`python/benchmarks/`**: Performance tracking
- **`r/`**: R reference implementation (Stage 7+)
- **`docs/`**: Documentation

## Adding New Pricing Models

1. Create `python/src/options_engine/models/my_model.py`
2. Implement pricing function:
   ```python
   def my_option_price(S, K, T, r, vol, **kwargs) -> float:
       """Price option; return scalar or array."""
       ...
   ```
3. Add tests in `python/tests/test_my_model.py`
4. Compare against BSM or known benchmarks
5. Add to `__init__.py` exports
6. Commit with clear message

## Adding New Greeks

1. Add to `python/src/options_engine/greeks/analytic.py` (if closed-form)
2. Or compute via finite differences in `numerical.py`
3. Validate against:
   - Known formulas (Hull, Haug)
   - Numerical finite differences (must agree within 1e-5)
   - Greek properties (e.g., vega ≥ 0)
4. Add tests

## Benchmarking

Use `python benchmarks/bench_*.py` to measure performance:
- Compare before/after optimization
- Report p50, p99 latency
- Report throughput (ops/sec)
- Commit results to `benchmarks/results.md`

## Code Style

- **Format**: Black (100-char line width)
- **Lint**: Ruff
- **Type hints**: Use `float | None` syntax (Python 3.10+)
- **Tests**: pytest with descriptive names
- **Docstrings**: One-liner + Args/Returns; no multi-line blocks

## Testing Standards

- ✅ Unit tests: Known values, edge cases, bounds
- ✅ Property tests: Greek relationships, monotonicity
- ✅ Numerical tests: Convergence, stability
- ✅ Cross-language: Python ↔ R reconciliation (Stage 7)

**Target coverage**: ≥85%

## Numerical Stability

When implementing new code:

1. **Cumulative normal**: Use `scipy.special.ndtr` (robust)
2. **Avoid cancellation**: Separate terms, e.g., theta = theta_vol + theta_discount
3. **Handle edge cases**: 0DTE, deep ITM/OTM, near-zero vol
4. **Validate inputs**: Raise clear errors for invalid (S, K, T, r, vol)
5. **Test extremes**: T=0, vol→0, rates<0, S>>K, S<<K

## Commit Messages

Format: `[Stage N] Feature: description`

Examples:
```
[Stage 1] BSM: add vanna Greek
[Stage 2] American: fix dividend handling at ex-dates
[Stage 3] MC: add control variate framework
[Bugfix] Numerics: robust cumulative normal clamping
```

## Pull Requests

1. Branch: `feature/description` or `fix/bug-name`
2. Keep PRs focused (one feature or fix per PR)
3. Add tests for all new code
4. Ensure all tests pass: `pytest python/tests/ -v`
5. Update CHANGELOG.md with changes
6. Link issues if fixing bugs

## Performance Targets

| Component | Target | Status |
|-----------|--------|--------|
| Single BSM option | <0.1 ms | ✅ ~0.03 ms |
| Vectorized (1000 opts) | <1 ms | ✅ ~0.1 ms |
| Full chain pricing | <5 ms | 🔜 Stage 5 |
| Implied vol solver | <1 ms | 🔜 Stage 4 |

## Questions?

See README.md or review existing tests for examples.

---

**Next Stages**: IV Solver (4), Chain Pricing (5), Real-Time Feed (6), R Cross-Check (7), Optimization (8), Repo Scaffolding (9)

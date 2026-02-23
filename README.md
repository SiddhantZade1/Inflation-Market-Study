# Inflation Regimes & Market Returns

> *Not all inflation is the same — and your hedge shouldn't be either.*

A quantitative study classifying 60+ years of US inflation into four distinct regimes and measuring how equities, bonds, gold, REITs, commodities, and TIPS perform **within each regime** — using both real (inflation-adjusted) returns and risk-adjusted metrics.

---

## Key Finding

Gold is widely considered the go-to inflation hedge. **The data shows it only reliably outperforms in one of four inflation regimes.**

TIPS, commodities, REITs, and equities each tell an equally nuanced story — one that gets obscured when you average across the full cycle.

---

## The Four Regimes

| Regime | CPI Level | Momentum | Historical Example |
|--------|-----------|----------|-------------------|
| **Low & Stable** | < 2.5% | Flat | 2010–2020 |
| **Rising** | Any | Accelerating | 2021–2022 |
| **High & Stable** | > 4.0% | Flat | 1974–1975, 1980 |
| **Falling** | > 2.5% | Decelerating | 1982–1984, 2022–2023 |

Regime momentum (6-month change in CPI YoY) captures the *direction* of inflation, which often drives asset prices more than the absolute level.

---

## Methodology

### 1. Data
- **Macro / inflation series** — FRED API (1962–present): CPI components, PCE, fed funds rate, yield curve, M2, breakeven inflation
- **Asset prices** — Yahoo Finance via yfinance: 7 asset class ETFs (SPY, IEF, TIP, GLD, VNQ, DJP, BIL)
- **Sector prices** — SPDR sector ETFs (XLE, XLK, XLF, XLU, XLI, XLB, XLP, XLY, XLV, XLRE)
- Monthly frequency throughout; real returns via Fisher equation

### 2. Regime Classification — Two Methods

**Threshold-based (primary):** Rule-based classifier using CPI level + 6-month momentum. Fast, interpretable, and easy to reproduce.

**Hidden Markov Model (validation):** GaussianHMM fitted on CPI YoY, 3-month annualised CPI, and real fed funds rate. Latent states sorted by CPI mean to match threshold labels. Agreement measured via Adjusted Rand Index.

### 3. Asset Class Analysis
- Mean/median monthly returns per asset per regime
- Annualised Sharpe ratios (regime-conditional)
- Real (inflation-adjusted) returns via Fisher equation
- % positive months as a reliability measure

### 4. Sector Hedge Analysis
- Rolling 24-month correlation: sector return vs CPI YoY
- Inflation beta: OLS regression of sector return on monthly CPI change
- Composite hedge score: Sharpe + correlation + win-rate during high-inflation regimes

---

## Project Structure

```
inflation-market-study/
├── README.md
├── requirements.txt
├── notebooks/
│   └── analysis.ipynb          ← main artifact, start here
├── src/
│   ├── __init__.py
│   ├── data_fetch.py           ← FRED + yfinance data pipeline
│   ├── regimes.py              ← regime classifier + visualisations
│   └── analysis.py             ← asset & sector analysis
└── data/
    ├── raw/                    ← cached CSVs (auto-generated, gitignored)
    └── processed/              ← processed outputs
```

---

## Quickstart

### 1. Clone & install dependencies
```bash
git clone https://github.com/YOUR_USERNAME/inflation-market-study.git
cd inflation-market-study
pip install -r requirements.txt
```

### 2. Get a FRED API key
Register for free at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)

```bash
export FRED_API_KEY=your_key_here
```

### 3. Run the notebook
```bash
cd notebooks
jupyter lab analysis.ipynb
```
Data is fetched on first run and cached to `data/raw/`. Subsequent runs load from cache.

---

## Usage as a Library

```python
from src import DataFetcher, RegimeClassifier, AssetClassAnalysis, SectorHedgeAnalysis

# Fetch data
fetcher = DataFetcher(fred_api_key='YOUR_KEY')
macro   = fetcher.get_macro()
master  = fetcher.get_master()

# Classify regimes
rc            = RegimeClassifier(macro)
macro_regimes = rc.classify_threshold()

# Validate with HMM
macro_hmm = rc.classify_hmm(n_states=4)

# Visualise
fig = rc.plot_regimes(df=macro_regimes)
fig.show()

# Asset class analysis
aca = AssetClassAnalysis(master.join(macro_regimes[['regime_label']]))
fig = aca.plot_regime_returns(metric='sharpe')
fig.show()

# Sector hedging
sha = SectorHedgeAnalysis(master.join(macro_regimes[['regime_label']]))
fig = sha.plot_hedge_scorecard()
fig.show()
```

---

## Data Sources

| Series | Source | Description |
|--------|--------|-------------|
| CPIAUCSL | FRED | CPI All Items (SA) |
| CPILFESL | FRED | CPI ex Food & Energy (Core) |
| T10YIE | FRED | 10-Year Breakeven Inflation Rate |
| FEDFUNDS | FRED | Effective Fed Funds Rate |
| DFII10 | FRED | 10-Year TIPS Yield (Real) |
| SPY, GLD, TIP, VNQ, DJP, IEF, BIL | Yahoo Finance | Asset class ETF proxies |
| XLE, XLK, XLF, XLU, XLI, XLB, XLP, XLY, XLV, XLRE | Yahoo Finance | SPDR sector ETFs |

Full series catalogue in `src/data_fetch.py`.

---

## Limitations

- ETF-based return data limits the asset analysis to roughly 2003–present. The macro/regime analysis extends to 1962.
- Regime boundaries are sensitive to threshold parameters. The HMM validation step tests robustness.
- All returns are pre-tax, pre-cost, and assume no rebalancing.
- US-only. International comparison (Japan, UK, EM) is a natural extension.
- ETFs don't perfectly replicate underlying assets (tracking error, expense ratios).

---

## Extensions (Potential Next Steps)

- [ ] International comparison (Japan, UK, Brazil — different inflation histories)
- [ ] Regime-aware portfolio optimisation (mean-variance with regime-conditional inputs)
- [ ] Out-of-sample backtest: trade based on real-time regime signal
- [ ] Nowcast-based inflation surprise signal (Cleveland Fed vs actual CPI)

---

## Dependencies

See `requirements.txt`. Core stack: `pandas`, `numpy`, `fredapi`, `yfinance`, `hmmlearn`, `statsmodels`, `scipy`, `plotly`, `scikit-learn`.

---

## License

MIT

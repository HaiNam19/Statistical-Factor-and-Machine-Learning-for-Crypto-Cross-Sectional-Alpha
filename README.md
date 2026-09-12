**Read this in other languages:** [English](README.md) | [Tiếng Việt](README.vi.md)

# CROSS-SECTIONAL FACTOR INVESTING FOR CRYPTO MARKET

This is a research report on cross-sectional factor investing for the Crypto market — building a long-short strategy based on ranking coins by quantitative characteristics (factors) instead of predicting absolute prices.

The final goal of this project is present that the whole process — from choosing the universe → creating factors → statistical testing → building alpha → backtesting — is carried out with the discipline of a professional quantitative research process: separating the discovery phase from the out-of-sample validation phase, avoiding look-ahead bias, and quantifying real transaction costs.

The pipeline has 4 notebooks (corresponding to 4 step files in src), executed in order and dependent on each other's data:

[1] Universe + Raw Factors  →  [2] Statistical Validation  →  [3] Composite Alpha  →  [4] Portfolio & Backtest

## Project Structure

```
crypto-quant-alpha/
├── notebook_analysis/    # 4 notebooks: universe, validation, alpha, backtest
├── src/                  # Reusable modules (features, validation, alpha, backtest)
├── data/                 # Raw OHLCV + processed parquet outputs
├── run_pipeline.py       # Entry point to run all 4 steps
├── requirements.txt
└── README.md
```

|  | Notebook and corresponding file                       | Stage             | Input                                                  | Output                                                                                           |
| - | ------------------------------ | --------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| 1 | `1_universe_factor_construction.ipynb` (`step1_feature.py`)  | Universe & Raw Factor | Raw market data                                 | Clean, point-in-time universe & raw factors, split into 2020–2021 (In Sample) and 2022–2025 (Out of Sample) |
| 2 | `2_statistical_significant_factor.ipynb` (`step2_validation.py`) | Statistical Filtering | Raw factors — **2020–2021 (IS)**                       | Factors with statistical evidence; does not use 2022–2025 data                           |
| 3 | `3_composite_alpha_construction.ipynb` (`step3_alpha.py`)      | Weighting Scheme      | Filtered factors + data **2022–2025**              | Weighting method using walk-forward: optimize weights on the past — apply to the future — using a rolling window that moves forward and repeats to combine 2 factors |
| 4 | `4_backtesting.ipynb` (`step4_backtest.py`)       | Execution & Cost      | Alpha + weighting scheme + data **2022–2025 (OOS)** | Executable portfolio, transaction costs, turnover and OOS performance                           |


### Main Results

| Metric           |    Value |
| ---------------- | -------: |
| `n_days`         |     1096 |
| `mean_daily_ret` | 0.001910 |
| `total_return`   | 4.482628 |
| `ann_return`     | 0.617407 |
| `ann_vol`        | 0.423536 |
| `sharpe`         | 1.457746 |
| `sortino`        | 2.163329 |
| `max_drawdown`   | 0.381515 |
| `win_rate`       | 0.563869 |
| `profit_factor`  | 1.209725 |
| `nw_t_stat`      | 2.542229 |
| `p_value`        | 0.011015 |

A few main comments:
  - Backtest results over 1,096 days show that the strategy has relatively good performance but comes with significant risk. The strategy reached a total return of about 448.3%, annualized return of 61.7%, while annualized volatility was 42.4%. From that, the Sharpe ratio reached 1.46 and the Sortino ratio reached 2.16
  - The biggest risk appears in the maximum drawdown of 38.15%. Even though annualized return reached 61.7%, the portfolio once lost about 38% from peak to trough before recovering. After a 38.15% drawdown, the portfolio needs to rise about 61.7% to return to its initial capital level. Therefore, Sharpe 1.46 should not be considered independently from drawdown.

<Figure size 640x480 with 1 Axes><img width="630" height="469" alt="image" src="https://github.com/user-attachments/assets/5366dc90-152f-4b8b-b9bf-ef72b6a436bb" />


Comments from the 12-month rolling Sharpe chart:

  - ~Oct 2023 – Oct 2024 (~12 months): rolling Sharpe fluctuated low, mostly below 1.0, with 2 clear bottoms — one around mid-2024 (~0.6) and one deepest bottom of the whole series around Oct 2024 (~0.2).
  - ~Nov 2024 – Jan 2026 (~14 months): Sharpe jumped up decisively, reached a peak of ~2.7 around March 2025, then fluctuated stably in the range 1.9–2.6 until the end of the sample.
 
 <Figure size 1200x500 with 2 Axes><img width="1102" height="490" alt="image" src="https://github.com/user-attachments/assets/fc0152da-c148-4248-a59e-69265f2145bd" />


From that, we can see that this is not an alpha that is stable and spread evenly over time, but a strategy with strong market-dependent effectiveness, where almost all the value was created in the last ~14 months. Monthly returns fluctuated a lot, and the drawdowns were clearly larger than the positive return events. At the same time, the results also match and explain why only 2 factors related to volatility survived the statistical tests.

**Final conclusion:** It is very likely that this strategy is benefiting from a specific market event rather than being a long-term sustainable strategy (partly because the statistical testing only runs from early 2020 to late 2021, so the data is not long enough).

### How to Run

#### 1. Clone the repo

```bash
git clone https://github.com/HaiNam19/Statistical-Factor-and-Machine-Learning-for-Crypto-Cross-Sectional-Alpha.git
cd Statistical-Factor-and-Machine-Learning-for-Crypto-Cross-Sectional-Alpha
```

#### 2. Create virtual environment

```bash
python -m venv .venv
```

#### 3. Activate virtual environment

Windows (CMD):

```cmd
.venv\Scripts\activate.bat
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

#### 4. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

Check that the installation succeeded:

```bash
python -c "from src.step1_features import Step1Pipeline; print('OK')"
```

Expected result:

```text
OK
```

#### 5. Prepare data and run

Place the raw file `ohlcv_daily.parquet` into the folder `data/universe/`.

Check that the file is in the right place (Windows):

```cmd
dir data\universe\ohlcv_daily.parquet
```

Run the whole pipeline:

```bash
python run_pipeline.py
```

Or run each step:

```bash
python run_pipeline.py --steps 1
python run_pipeline.py --steps 2
python run_pipeline.py --steps 3
python run_pipeline.py --steps 4
```

The output files are saved into `data/processed/`:

```text
B1_factor_construction.parquet
2022-2025_factor_construction.parquet
eligible_factors.parquet
D1_df_weightt.parquet
backtest_plot.png
```

## Module Description

### I. 1_universe_factor_construction — Universe & Factor Construction

Goal: Build the investment universe for each day (point-in-time) and create raw factors for each coin in the universe at that moment.

Input: ohlcv_daily.parquet — daily OHLCV data from 2020-01-01 to 2026-08-17.


Universe Construction:
  - Require at least 6 months of trading history at time t.
  - Remove stablecoins, leveraged/wrapped tokens and invalid tokens.
  - Choose Top <= 40 by median_dollar_volume_30d, with a minimum liquidity of $10M/day.

Raw Factors — 4 groups:
  - Momentum: 7D / 14D / 30D / 90D
  - Reversal: 1D / 3D
  - Volatility: 7D / 14D / 30D + Vol-of-Vol 14D
  - Liquidity: Amihud 14D / 30D
    
Factors are winsorized (5–95%) and cross-sectional z-scored by day inside the universe. Forward returns are computed for 1D / 5D / 14D.
Data Split:
  - 2020–2021: Factor discovery → Notebook 2
  - 2022–2025: Weight fitting & OOS evaluation → Notebooks 3–4


### II. 2_statistical_significant_factor.ipynb — Statistical Validation

Goal: Test 12 raw factors and keep only the factors that have statistical evidence and economic signal.

Input: B1_factor_construction.parquet — period 2020–2021

Statistical Tests:
  - Daily Spearman IC: Factor vs. forward return, test the mean IC using Newey-West HAC.
  - Univariate Fama-MacBeth: Cross-sectional factor return and NW t-stat.
  - Tertile Portfolio Spread: Buy the high-factor group, sell the low-factor group, measure spread in bps/day.

Factor selection rules:
  - At least 1 out of 3 tests is significant at the 5% level.
  - IC / beta / spread have consistent signs.
  - Spread ≥ 10 bps/day so it can potentially cover transaction costs.
  - Check cross-factor correlation and remove redundant factors.

Results:
  - 2 out of 12 factors are kept: z_vol_14d and z_vol_of_vol_14d.
  - Momentum, reversal and Amihud illiquidity are all removed.
  - Both selected factors belong to the volatility group.

**Limitation:** The 2020–2021 period is only about 15 months long and carries the characteristics of a distinct market regime (COVID), so the factor selection results may not yet represent other market regimes. I will make improvements in later versions, focusing more on evaluating each period.

### III. 3_composite_alpha_construction.ipynb — Composite Alpha Construction

Goal: Combine the 2 selected factors (z_vol_14d, z_vol_of_vol_14d) into one composite alpha score and compare different weighting methods.

Input: 2022-2025_factor_construction.parquet — data completely separated from the factor selection phase in Notebook 2.

Method:
  - Walk-forward CV: Rolling 12-month train → 1-month test, every month from 2023-01 to 2025-12.
  - Compare 4 methods:
      - Equal-weight: Arithmetic mean of the 2 factor z-scores.
      - Evidence-weight: Weight based on ic_mean over each training window.
      - Ridge Regression: Linear model with regularization to predict forward return.
      - LightGBM: Gradient boosting similar to Ridge to predict forward return.
  - Results:
      - Equal-weight is the method chosen as the main weight.
      - Ridge and LightGBM do not outperform equal-weight.
      - Evidence-weight on average outperforms equal-weight but is not statistically significant.
        
**Decision:** Only keep equal-weight to avoid overfitting and to ensure consistency between alpha construction and the downstream backtest.

### IV. 4_backtesting.ipynb — Backtest

Goal: Turn the composite alpha (equal-weight) into an executable long/short portfolio, simulate transaction costs, and evaluate performance.

Input:
  - 2022-2025_factor_construction.parquet — uses z_amihud_14d for the cost model.
  - D1_df_weightt.parquet — composite weights from Notebook 3

Method:
  - Portfolio Construction: Split the universe into 3 groups by composite score; long the top tertile, short the bottom tertile, equal-weight inside each leg.
  - Transaction Costs: Slippage is tiered by liquidity (z_amihud_14d) with 4 buckets: 2 / 5 / 10 / 20 bps, plus a fixed cost by turnover.
  - Performance: Compute gross/net returns and metrics: Annualized Return, Volatility, Sharpe, Sortino, Max Drawdown, Win Rate, Profit Factor and Newey-West t-stat.

Results:
  - The OOS backtest 2023–2025 only uses equal-weight, consistent with the decision from Notebook 3.
  - Mean daily return is statistically significant at the 5% level according to Newey-West.

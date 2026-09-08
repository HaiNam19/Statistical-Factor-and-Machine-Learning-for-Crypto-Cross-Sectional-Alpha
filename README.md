**Read in another language:** [English](README.md) | [Vietnamese](README.vi.md)

# CROSS-SECTIONAL FACTOR INVESTING FOR CRYPTO MARKET

This is a research report on investing based on cross-sectional factors in the Crypto market - Building a long-short strategy based on ranking coins by quantitative characteristics (factors) instead of predicting absolute prices.

The final goal of the project is not to maximize a good-looking Sharpe ratio, but to prove whether the entire process from universe selection → factor construction → statistical validation → alpha construction → backtesting follows the discipline of a professional quantitative research process: separating the exploration period from the out-of-sample testing period, avoiding look-ahead bias, and measuring realistic transaction costs.

The pipeline consists of 4 notebooks, executed sequentially and dependent on each other's data:

[1] Universe + Raw Factors  →  [2] Statistical Validation  →  [3] Composite Alpha  →  [4] Portfolio & Backtest

## Project Structure

|   | Notebook                       | Stage                 | Input                                               | Output                                                                                                                                               |
| - | ------------------------------ | --------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | `1_Universe_RawFactors.ipynb`  | Universe & Raw Factor | Raw market data                                     | Clean, point-in-time universe & raw factors, split into 2020–2021 (In Sample) and 2022–2025 (Out of Sample)                                          |
| 2 | `2_StatisticalFiltering.ipynb` | Statistical Filtering | Raw factors — **2020–2021 (IS)**                    | Factors with statistical evidence; 2022–2025 data is not used                                                                                        |
| 3 | `3_WeightingScheme.ipynb`      | Weighting Scheme      | Selected factors + data **2022–2025**               | Walk-forward weighting method: optimize weights on the past - apply them to the future - move the window forward and repeat to combine the 2 factors |
| 4 | `4_Execution_Cost.ipynb`       | Execution & Cost      | Alpha + weighting scheme + data **2022–2025 (OOS)** | Executable portfolio, transaction costs, turnover, and OOS performance                                                                               |

## Module Descriptions

### I. 1_universe_factor_construction — Universe & Factor Construction

Objective: Build the investment universe for each day (**point-in-time**) and create raw factors for each coin in the universe at that time.

Input: `ohlcv_daily.parquet` — daily OHLCV data from 2020-01-01 to 2026-08-17.

Universe Construction:

  * Require at least 6 months of trading history at time t.
  * Remove stablecoins, leveraged/wrapped tokens, and invalid tokens.
  * Select Top ≤40 by median_dollar_volume_30d, with minimum liquidity of $10M/day.

Raw Factors — 4 groups:

  * Momentum: 7D / 14D / 30D / 90D
  * Reversal: 1D / 3D
  * Volatility: 7D / 14D / 30D + Vol-of-Vol 14D
  * Liquidity: Amihud 14D / 30D

Factors are winsorized (5–95%) and cross-sectional z-scored by day within the universe. Forward returns are calculated for 1D / 5D / 14D.

Data Split:

  * 2020–2021: Factor discovery → Notebook 2
  * 2022–2025: Weight fitting & OOS evaluation → Notebooks 3–4

### II. 2_statistical_significant_factor.ipynb — Statistical Validation

Objective: Test the 12 raw factors and keep only factors with statistical evidence and economic signal.

Input: B1_factor_construction.parquet — 2020–2021 period

Statistical Tests:

  * Daily Spearman IC: Factor vs. forward return, with mean IC tested using Newey-West HAC.
  * Univariate Fama-MacBeth: Cross-sectional factor return and NW t-stat.
  * Tertile Portfolio Spread: Buy the high-factor group, sell the low-factor group, and measure the spread in bps/day.

Factor selection rules:

  * At least 1/3 tests are significant at the 5% level.
  * C / beta / spread have consistent signs.
  * Spread ≥ 10 bps/day to have the potential to cover transaction costs.
  * Check cross-factor correlation and remove redundant factors.

Results:

  * 2/12 factors are retained: z_vol_14d and z_vol_of_vol_14d.
  * Momentum, reversal, and Amihud illiquidity are removed.
  * Both selected factors belong to the volatility group.

**Limitation:** The 2020–2021 period covers only about 15 months and has a specific market regime (COVID), so the factor selection results may not represent other market regimes. I will make improvements in later versions, focusing more on evaluation across different periods.

### III. 3_composite_alpha_construction.ipynb — Composite Alpha Construction

Objective: Combine the 2 selected factors (z_vol_14d, z_vol_of_vol_14d) into a composite alpha score and compare different weighting methods.

Input: 2022-2025_factor_construction.parquet — data completely separated from the factor selection period in Notebook 2.

Methodology:

  * Walk-forward CV: Rolling 12-month train → 1-month test, monthly from 2023-01 to 2025-12.
  * Compare 4 methods:

    * Equal-weight: Simple average of the 2 factor z-scores.
    * Evidence-weight: Weights based on ic_mean from each training window.
    * Ridge Regression: Linear model with regularization to predict forward return.
    * LightGBM: Gradient boosting similar to Ridge to predict forward return

  * Results:
  
    * Equal-weight is selected as the main weighting method.
    * Ridge and LightGBM do not outperform equal-weight.
    * Evidence-weight performs better than equal-weight on average, but the difference is not statistically significant.

**Decision:** Keep only equal-weight to avoid overfitting and ensure consistency between alpha construction and downstream backtesting.

### IV. 4_backtesting.ipynb — Backtest

Objective: Convert the composite alpha (equal-weight) into an executable long/short portfolio, simulate transaction costs, and evaluate performance.

Input:

  * 2022-2025_factor_construction.parquet — uses z_amihud_14d for the cost model.
  * D1_df_weightt.parquet — composite weights from Notebook 3

Methodology:

  * Portfolio Construction: Divide the universe into 3 groups based on the composite score; long the top tertile, short the bottom tertile, equal-weight within each leg
  * Transaction Costs: Slippage based on liquidity (z_amihud_14d) with 4 buckets: 2 / 5 / 10 / 20 bps, plus fixed cost based on turnover.
  * Performance: Calculate gross/net returns and the following metrics: Annualized Return, Volatility, Sharpe, Sortino, Max Drawdown, Win Rate, Profit Factor and Newey-West t-stat.

Results:

  * OOS backtest 2023–2025 uses only equal-weight, consistent with the decision from Notebook 3.
  * Mean daily return is statistically significant at the 5% level according to the Newey-West test.

"""Bước 4: Portfolio construction → Gross return → Cost → Net performance.

Tương ứng notebook 4 (4_backtesting.ipynb).
"""
import logging

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS

from . import config

logger = logging.getLogger(__name__)


class PortfolioConstructor:
    """Long top / short bottom theo tertile của weight."""

    def __init__(self, n_groups=None):
        self.n_groups = n_groups or config.N_GROUPS

    def construct(self, df: pd.DataFrame, weight_col: str) -> dict:
        temp = df.copy().sort_values(
            ["timestamp", weight_col], ascending=[True, False]
        )
        weights_list = []
        daily_metrics = []

        for date, group in temp.groupby("timestamp"):
            group = group.dropna(subset=[weight_col]).copy()
            n_coins = len(group)
            if n_coins < self.n_groups:
                continue

            group["rank"] = group[weight_col].rank(method="first", ascending=False)
            group["group"] = pd.qcut(
                group["rank"], q=self.n_groups,
                labels=list(range(self.n_groups)), duplicates="drop",
            ).astype(int)

            n_top = (group["group"] == 0).sum()
            n_bottom = (group["group"] == self.n_groups - 1).sum()

            group["weight"] = 0.0
            if n_top > 0:
                group.loc[group["group"] == 0, "weight"] = 1.0 / n_top
            if n_bottom > 0:
                group.loc[group["group"] == self.n_groups - 1, "weight"] = -1.0 / n_bottom

            weights_list.append(
                group[["timestamp", "symbol", "weight", "fwd_1d"]].copy()
            )
            daily_metrics.append({
                "timestamp": date,
                "n_long": n_top,
                "n_short": n_bottom,
                "gross_exposure": group["weight"].abs().sum(),
                "net_exposure": group["weight"].sum(),
                "n_coins": n_coins,
            })

        weights_df = pd.concat(weights_list, ignore_index=True)
        daily_summary = pd.DataFrame(daily_metrics)

        pivot = weights_df.pivot(index="symbol", columns="timestamp",
                                values="weight").fillna(0)
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)
        turnover = 0.5 * pivot.diff(axis=1).abs().sum(axis=0)
        turnover.name = "turnover"

        daily_summary = daily_summary.merge(
            turnover, left_on="timestamp", right_index=True, how="left"
        )
        daily_summary["turnover"] = daily_summary["turnover"].fillna(0.0)
        daily_summary = daily_summary.sort_values("timestamp").reset_index(drop=True)

        return {"weights_df": weights_df, "daily_summary": daily_summary}


class GrossReturnBuilder:
    """Tính gross return từ weight × fwd_1d."""

    @staticmethod
    def build(weights_df: pd.DataFrame) -> dict:
        temp = weights_df.copy()
        temp["contribution"] = temp["fwd_1d"] * temp["weight"]
        daily = temp.groupby("timestamp", as_index=False)["contribution"].sum()
        daily = daily.rename(columns={"contribution": "gross_return_daily"})
        return {"weights_df": temp, "gross_return_daily": daily}


class CostModel:
    """Slippage theo percentile của z_amihud_14d + fee bps cố định."""

    def __init__(self, fee_bps=None, buckets=None):
        self.fee_bps = fee_bps or config.FEE_BPS
        self.buckets = buckets or config.SLIPPAGE_BUCKETS

    def compute(self, weights_df: pd.DataFrame,
                oot_df: pd.DataFrame) -> pd.DataFrame:
        temp = weights_df.merge(
            oot_df[["timestamp", "symbol", "z_amihud_14d"]],
            on=["timestamp", "symbol"], how="inner",
        )

        # Percentile rank theo amihud trong ngày
        ranks, pct = [], []
        for _, g in temp.groupby("timestamp"):
            n = len(g)
            r = g["z_amihud_14d"].rank(method="first", ascending=True)
            ranks.append(r)
            pct.append(r / n * 100)
        temp["rank_t_day"] = pd.concat(ranks)
        temp["percentile_t_day"] = pd.concat(pct)

        # Bucket
        bucket_list = []
        for _, g in temp.groupby("timestamp"):
            b1 = g["percentile_t_day"].quantile(0.5)
            b2 = g["percentile_t_day"].quantile(0.8)
            b3 = g["percentile_t_day"].quantile(0.95)
            for p in g["percentile_t_day"]:
                if p <= b1:
                    bucket_list.append(1)
                elif p <= b2:
                    bucket_list.append(2)
                elif p <= b3:
                    bucket_list.append(3)
                else:
                    bucket_list.append(4)
        temp["bucket"] = bucket_list
        temp["slippage_bps"] = temp["bucket"].map(self.buckets)

        # Trade = |delta weight|
        pivot = temp.pivot(index="symbol", columns="timestamp",
                          values="weight").fillna(0)
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)
        trades = pivot.diff(axis=1).abs().stack().reset_index()
        trades.columns = ["symbol", "timestamp", "trade"]

        temp = temp.merge(trades, on=["timestamp", "symbol"], how="left")
        temp["trade"] = temp["trade"].fillna(0)
        temp["slippage_cost"] = np.where(
            temp["trade"] > 0, temp["slippage_bps"] * temp["trade"], 0.0
        )

        daily = temp.groupby("timestamp").agg(
            slippage_cost_day_t=("slippage_cost", "sum"),
            total_trade=("trade", "sum"),
        ).reset_index()
        daily["turnover"] = 0.5 * daily["total_trade"]
        daily["fee_cost"] = daily["turnover"] * self.fee_bps
        daily["total_cost"] = daily["slippage_cost_day_t"] + daily["fee_cost"]

        return daily[["timestamp", "turnover", "fee_cost",
                      "slippage_cost_day_t", "total_cost"]]


class PerformanceAnalyzer:
    """Metric hiệu suất + plot."""

    @staticmethod
    def net_performance(gross_daily: pd.DataFrame,
                       cost_daily: pd.DataFrame) -> pd.DataFrame:
        df = gross_daily.merge(cost_daily, on="timestamp", how="inner")
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["net_return"] = df["gross_return_daily"] - df["total_cost"] / 10000
        df["cumulative"] = (1 + df["net_return"]).cumprod() - 1
        return df

    @staticmethod
    def summary(net_df: pd.DataFrame) -> pd.DataFrame:
        net = net_df["net_return"].dropna().values
        n = len(net)
        mean_ret = np.mean(net)
        std_ret = np.std(net)
        total_ret = net_df["cumulative"].iloc[-1]
        ann_ret = (1 + mean_ret) ** 252 - 1
        ann_vol = std_ret * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

        neg = net[net < 0]
        downside = np.std(neg) * np.sqrt(252) if len(neg) > 0 else np.nan
        sortino = ann_ret / downside if downside and downside > 0 else np.nan

        nav = 1 + net_df["cumulative"].values
        peak = np.maximum.accumulate(nav)
        max_dd = ((peak - nav) / peak).max()

        pos = net[net > 0]
        win_rate = len(pos) / n
        profit_factor = pos.sum() / abs(neg.sum()) if neg.sum() != 0 else np.inf

        X = np.ones((n, 1))
        try:
            model = OLS(net, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
            nw_t = model.tvalues[0]
            p = model.pvalues[0]
        except Exception:
            model = OLS(net, X).fit()
            nw_t = model.tvalues[0]
            p = model.pvalues[0]

        return pd.DataFrame({
            "metric": ["n_days", "mean_daily_ret", "total_return", "ann_return",
                       "ann_vol", "sharpe", "sortino", "max_drawdown",
                       "win_rate", "profit_factor", "nw_t_stat", "p_value"],
            "value": [n, mean_ret, total_ret, ann_ret, ann_vol, sharpe, sortino,
                      max_dd, win_rate, profit_factor, nw_t, p],
        })

    @staticmethod
    def rolling_sharpe(net_df: pd.DataFrame, window: int = 365) -> pd.DataFrame:
        df = net_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").drop_duplicates("timestamp")
        df = df.set_index("timestamp")
        min_periods = int(0.8 * window)
        mean = df["net_return"].rolling(window=window, min_periods=min_periods).mean()
        std = df["net_return"].rolling(window=window, min_periods=min_periods).std()
        df["rolling_sharpe"] = mean / std * np.sqrt(window)
        return df

    @staticmethod
    def plot_cumulative(net_df: pd.DataFrame, ax=None, label="strategy"):
        ax = ax or plt.gca()
        ax.plot(net_df["timestamp"], net_df["cumulative"], label=label)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
        )
        ax.set_title("Cumulative net return")
        ax.set_xlabel("time")
        ax.set_ylabel("cumulative")
        ax.legend()
        ax.grid()
        return ax

    @staticmethod
    def plot_rolling_sharpe(net_df: pd.DataFrame, window=365, ax=None, label="strategy"):
        df = PerformanceAnalyzer.rolling_sharpe(net_df, window)
        ax = ax or plt.gca()
        ax.plot(df.index, df["rolling_sharpe"], label=label)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
        )
        ax.set_title(f"Rolling Sharpe {window}d")
        ax.set_xlabel("time")
        ax.set_ylabel("sharpe")
        ax.legend()
        ax.grid()
        return ax
    
    @staticmethod
    def plot_monthly_heatmap(net_df: pd.DataFrame, ax = None):
        temp = net_df.copy()
        temp["year"] = temp["timestamp"].dt.year
        temp["month"] = temp["timestamp"].dt.month
        
        monthly = (temp.groupby(["year", "month"])["net_return"]
                    .apply(lambda x: (1 + x).prod() - 1).reset_index())
        pivot = monthly.pivot(index = "year", columns = "month", values = "net_return")
        pivot = pivot.reindex(columns=range(1,13))
        
        ax = ax or plt.gca()
        im = ax.imshow(pivot.values * 100, cmap="RdYlGn", aspect="auto",
                   vmin=-15, vmax=15)
        
        ax.set_xticks(range(12))
        ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                                "Jul","Aug","Sep","Oct","Nov","Dec"])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center",
                            fontsize=9,
                            color="black" if abs(val) < 0.08 else "white")
        
        plt.colorbar(im, ax=ax, label="Monthly return (%)")
        ax.set_title("Monthly Net Returns Heatmap (%)")
        ax.set_xlabel("Month")
        ax.set_ylabel("Year")
        return ax
        

class Step4Pipeline:

    def __init__(self, weight_col="equal_weight", n_groups=None):
        self.weight_col = weight_col
        self.constructor = PortfolioConstructor(n_groups)

    def run(self, df_weights: pd.DataFrame,
            oot_df: pd.DataFrame) -> dict:
        portfolio = self.constructor.construct(df_weights, self.weight_col)
        gross = GrossReturnBuilder.build(portfolio["weights_df"])
        cost = CostModel().compute(portfolio["weights_df"], oot_df)
        net = PerformanceAnalyzer.net_performance(
            gross["gross_return_daily"], cost
        )
        summary = PerformanceAnalyzer.summary(net)
        return {
            "portfolio": portfolio,
            "gross": gross,
            "cost": cost,
            "net": net,
            "summary": summary,
        }

    def plot(self, net_df: pd.DataFrame, save_path=None):
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        PerformanceAnalyzer.plot_cumulative(
            net_df, ax=axes[0], label=f"net ({self.weight_col})"
        )
        PerformanceAnalyzer.plot_rolling_sharpe(
            net_df, ax=axes[1], label=f"sharpe ({self.weight_col})"
        )
        PerformanceAnalyzer.plot_monthly_heatmap(
            net_df, ax = axes[2] 
        )
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.show()
        


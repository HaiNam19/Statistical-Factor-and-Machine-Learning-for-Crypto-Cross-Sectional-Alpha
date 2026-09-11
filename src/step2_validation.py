"""Bước 2: Factor validation + chọn factor đủ điều kiện.

Tương ứng notebook 2 (2_statistical_significant_factor.ipynb).

Gồm 4 kiểm định:
    - C1: Daily IC (Spearman)
    - C2: IC summary + Newey-West t-stat
    - C3: Fama-MacBeth univariate regression
    - C4: Tertile portfolio spread
    - C5: Correlation filter để loại factor trùng
"""
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr, t as t_dist

from . import config

logger = logging.getLogger(__name__)


class FactorValidator:

    def __init__(self, factor_cols, forward_cols, n_groups=3):
        self.factor_cols = factor_cols
        self.forward_cols = forward_cols
        self.n_groups = n_groups
        self.lag_map = config.NW_LAG_MAP

    @staticmethod
    def _sig_label(p) -> str:
        if pd.isna(p):
            return "not significant"
        if p < 0.01:
            return "alpha 1%"
        if p < 0.05:
            return "alpha 5%"
        return "not significant"

    @staticmethod
    def _nw_fit(y: np.ndarray, lag: int):
        n = len(y)
        if n < 2:
            return None
        lag = min(lag, n - 1)
        X = np.ones((n, 1))
        return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lag})

    def build_daily_ic(self, df: pd.DataFrame) -> pd.DataFrame:
        in_u = df[df["in_universe"] == 1]
        rows = []
        for date, group in in_u.groupby("timestamp"):
            for factor in self.factor_cols:
                for fwd in self.forward_cols:
                    corr, _ = spearmanr(group[factor], group[fwd])
                    rows.append({
                        "timestamp": date,
                        "factor_id": factor,
                        "horizon": fwd,
                        "ic_spearmanr": corr,
                    })
        return pd.DataFrame(rows)

    def build_ic_summary(self, daily_ic: pd.DataFrame) -> pd.DataFrame:
        stats = daily_ic.groupby(["factor_id", "horizon"]).agg(
            ic_mean=("ic_spearmanr", "mean"),
            ic_std=("ic_spearmanr", "std"),
        ).reset_index()
        stats["ic_ir"] = stats["ic_mean"] / stats["ic_std"]

        nw_rows = []
        for (factor, fwd), g in daily_ic.groupby(["factor_id", "horizon"]):
            ic = g["ic_spearmanr"].dropna().astype(float).values
            lag = self.lag_map.get(fwd, 0)
            m = self._nw_fit(ic, lag)
            nw_rows.append({
                "factor_id": factor, "horizon": fwd, "nw_lag": lag,
                "nw_se": m.bse[0] if m else np.nan,
                "nw_t_stat": m.tvalues[0] if m else np.nan,
                "n_days": len(ic),
            })
        out = stats.merge(pd.DataFrame(nw_rows), on=["factor_id", "horizon"], how="left")
        out["p_value"] = 2 * t_dist.sf(np.abs(out["nw_t_stat"]), out["n_days"] - 1)
        out["sig"] = out["p_value"].apply(self._sig_label)
        return out

    @staticmethod
    def _winsorize_fwd(df: pd.DataFrame) -> pd.DataFrame:
        t = df.copy()
        for c in [c for c in df.columns if c.startswith("fwd_")]:
            t[c] = t.groupby("timestamp")[c].transform(
                lambda x: x.clip(lower=x.quantile(0.05), upper=x.quantile(0.95))
            )
        return t

    def build_fama_macbeth(self, df: pd.DataFrame) -> pd.DataFrame:
        df_w = self._winsorize_fwd(df)
        in_u = df_w[df_w["in_universe"] == 1]
        rows = []
        for factor in self.factor_cols:
            for fwd in self.forward_cols:
                betas, alphas, n_obs = [], [], []
                for _, g in in_u.groupby("timestamp"):
                    g = g.dropna(subset=[factor, fwd])
                    if len(g) < 3:
                        continue
                    X = sm.add_constant(g[factor])
                    m = sm.OLS(g[fwd], X).fit()
                    betas.append(m.params.iloc[1])
                    alphas.append(m.params.iloc[0])
                    n_obs.append(len(g))

                betas = np.array(betas)
                if len(betas) < 2:
                    continue
                lag = self.lag_map.get(fwd, 0)
                m = self._nw_fit(betas, lag)
                p = 2 * t_dist.sf(abs(m.tvalues[0]), len(betas) - 1)

                rows.append({
                    "factor_id": factor, "horizon": fwd,
                    "beta_mean": betas.mean(), "beta_std": betas.std(),
                    "nw_se": m.bse[0], "nw_t_stat": m.tvalues[0],
                    "p_value": p, "alpha_mean": np.mean(alphas),
                    "n_days": len(betas), "avg_n_obs": np.mean(n_obs),
                    "sig": self._sig_label(p),
                })
        return pd.DataFrame(rows)

    def build_day_portfolio_spread(self, df: pd.DataFrame) -> pd.DataFrame:
        in_u = df[df["in_universe"] == 1]
        rows = []
        for date, g in in_u.groupby("timestamp"):
            g = g.dropna()
            if len(g) < self.n_groups:
                continue
            for factor in self.factor_cols:
                gs = g.sort_values(factor).copy()
                gs["group"] = pd.qcut(
                    np.arange(len(gs)), q=self.n_groups, labels=range(self.n_groups)
                )
                for fwd in self.forward_cols:
                    means = gs.groupby("group", observed=True)[fwd].mean()
                    rows.append({
                        "date": date, "factor_id": factor,
                        "forward_return": fwd,
                        "spread": means.iloc[-1] - means.iloc[0],
                        "avg_return_low": means.iloc[0],
                        "avg_return_mid": means.iloc[len(means) // 2],
                        "avg_return_high": means.iloc[-1],
                    })
        return pd.DataFrame(rows)

    def build_portfolio_spread(self, day_spread: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for (factor, fwd), g in day_spread.groupby(["factor_id", "forward_return"]):
            s = g.sort_values("date")["spread"].dropna()
            if len(s) < 2:
                continue
            lag = self.lag_map.get(fwd, 0)
            m = self._nw_fit(s.values, lag)
            p = 2 * t_dist.sf(abs(m.tvalues[0]), len(s) - 1)
            rows.append({
                "factor_id": factor, "horizon": fwd,
                "spread_mean": s.mean(), "spread_std": s.std(),
                "nw_se": m.bse[0], "nw_t_stat": m.tvalues[0],
                "p_value": p, "sig": self._sig_label(p),
            })
        return pd.DataFrame(rows)

    def build_summary_factor(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Running full factor validation ...")
        daily_ic = self.build_daily_ic(df)
        ic_sum = self.build_ic_summary(daily_ic)[
            ["factor_id", "horizon", "ic_mean", "sig"]
        ].rename(columns={"sig": "sig_y"})

        fm = self.build_fama_macbeth(df)[
            ["factor_id", "horizon", "beta_mean", "sig"]
        ].rename(columns={"sig": "sig_beta"})

        spread = self.build_day_portfolio_spread(df)
        spread_sum = self.build_portfolio_spread(spread)[
            ["factor_id", "horizon", "spread_mean", "sig"]
        ].rename(columns={"sig": "sig_x"})

        summary = (
            spread_sum.merge(ic_sum, on=["factor_id", "horizon"])
                      .merge(fm, on=["factor_id", "horizon"])
        )

        summary["spread_mean_magnitude"] = summary.apply(
            lambda r: abs(r["spread_mean"] * 10000) / self._horizon_days(r["horizon"]),
            axis=1,
        )
        summary["eligible"] = summary.apply(self._check_quality, axis=1)
        summary["sign_orientation"] = summary.apply(
            lambda r: 1 if (
                self._sign(r["spread_mean"]) + self._sign(r["ic_mean"])
                + self._sign(r["beta_mean"])
            ) > 0 else -1,
            axis=1,
        )
        return summary

    @staticmethod
    def _horizon_days(h: str) -> int:
        return {"fwd_1d": 1, "fwd_5d": 5, "fwd_14d": 14}.get(h, 1)

    @staticmethod
    def _sign(x):
        if pd.isna(x):
            return 0
        return 1 if x > 0 else (-1 if x < 0 else 0)

    def _check_quality(self, row) -> bool:
        sig_x = row["sig_x"] in config.SIG_LEVELS
        sig_y = row["sig_y"] in config.SIG_LEVELS
        sig_beta = row["sig_beta"] in config.SIG_LEVELS
        n_sig = sum([sig_x, sig_y, sig_beta])

        signs = [self._sign(row["spread_mean"]),
                 self._sign(row["ic_mean"]),
                 self._sign(row["beta_mean"])]
        sign_consistence = (sum(signs) == 3) or (sum(signs) == -3)

        enough = row["spread_mean_magnitude"] >= config.MIN_SPREAD_MAGNITUDE

        return n_sig >= 1 and sign_consistence and enough


class FactorSelector:
    """Chọn factor cuối cùng: loại factor trùng nhau (corr cao)."""

    def __init__(self, max_corr: float = 0.7):
        self.max_corr = max_corr

    def check_corr(self, df: pd.DataFrame, factor_cols) -> pd.DataFrame:
        in_u = df[df["in_universe"] == 1]
        daily_corr = (
            in_u.groupby("timestamp")[factor_cols]
            .corr(method="spearman")
            .rename_axis(["timestamp", "factor"])
        )
        median_corr = daily_corr.groupby(level="factor").median()
        upper = median_corr.where(
            np.triu(np.ones(median_corr.shape), k=1).astype(bool)
        )
        pairs = upper.stack().reset_index()
        pairs.columns = ["factor_a", "factor_b", "corr"]
        return pairs.dropna()

    def select_final_factors(self, summary: pd.DataFrame,
                             corr_pairs: pd.DataFrame,
                             horizon: str = "fwd_1d") -> list:
        """Trả về list factor cuối cùng sau khi bỏ factor trùng."""
        eligible_at_h = summary[
            (summary["eligible"]) & (summary["horizon"] == horizon)
        ]
        factor_ids = eligible_at_h["factor_id"].unique().tolist()

        # Quality score: |ic_mean| lớn nhất ở horizon đang xét
        quality = eligible_at_h.groupby("factor_id")["ic_mean"].apply(
            lambda x: x.abs().max()
        ).to_dict()

        drop = set()
        for _, row in corr_pairs.iterrows():
            a, b, c = row["factor_a"], row["factor_b"], row["corr"]
            if c <= self.max_corr or a not in factor_ids or b not in factor_ids:
                continue
            if a in drop or b in drop:
                continue
            loser = a if quality.get(a, 0) < quality.get(b, 0) else b
            drop.add(loser)

        return [f for f in factor_ids if f not in drop]

class Step2Pipeline:

    def __init__(self):
        self.validator = FactorValidator(
        config.ALL_FACTOR_COLS_Z, config.ALL_FORWARD_COLS
    )
        self.selector = FactorSelector()
        
        
    def run(self, df: pd.DataFrame, horizon: str = "fwd_1d") -> dict:
        df = df[df["timestamp"] >= config.IN_SAMPLE_START].copy()
        summary = self.validator.build_summary_factor(df)
        eligible = summary[summary["eligible"]]

        # CHỈ giữ factor pass ở horizon mong muốn (mặc định fwd_1d)
        eligible_at_h = eligible[eligible["horizon"] == horizon]
        factor_ids = eligible_at_h["factor_id"].unique().tolist()
        logger.info("Eligible factors at %s: %s", horizon, factor_ids)

        corr_pairs = self.selector.check_corr(df, factor_ids)
        final_factors = self.selector.select_final_factors(summary, corr_pairs)

        # Đổi tên cột cho dễ đọc (giống notebook)
        eligible_at_h = eligible_at_h.rename(columns={
            "sig_x": "sig_x", "sig_y": "sig_y", "sig": "sig_beta"
        })

        logger.info("Final factors after corr filter: %s", final_factors)
        return {
            "summary": summary,
            "eligible": eligible_at_h,           # chỉ fwd_1d
            "correlation_pairs": corr_pairs,
            "final_factors": final_factors,
        }
"""Bước 3: Composite Alpha — equal / evidence / ridge / lightgbm.

Tương ứng notebook 3 (3_composite_alpha_construction.ipynb).

Output: D1_df_weightt.parquet chứa 4 loại weight trên OOT 2023-2025.
"""
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from . import config

logger = logging.getLogger(__name__)


class WalkForwardSplitter:
    """Tạo fold walk-forward: train 12 tháng → test 1 tháng."""

    def __init__(self, train_months=None, test_window=None,
                 research_start=None, research_end=None, data_start=None):
        self.train_months = train_months or config.TRAIN_WINDOW_MONTHS
        self.research_start = pd.to_datetime(
            research_start or config.OOT_START, utc=True
        )
        self.research_end = pd.to_datetime(
            research_end or config.OOT_END, utc=True
        )
        self.data_start = pd.to_datetime(data_start or config.DATA_START, utc=True)

    def build(self) -> pd.DataFrame:
        test_months = pd.date_range(
            start=self.research_start, end=self.research_end, freq="MS"
        )
        folds = []
        for i, test_start in enumerate(test_months):
            # Đảm bảo tz-aware UTC (date_range có thể trả về tz-aware nếu input tz-aware)
            if test_start.tz is None:
                test_start = test_start.tz_localize("UTC")
            else:
                test_start = test_start.tz_convert("UTC")

            test_end = test_start + pd.offsets.MonthEnd(0)
            train_end = test_start - pd.Timedelta(days=1)
            train_start = (train_end - pd.DateOffset(
                months=self.train_months - 1)).replace(day=1)

            if train_start < self.data_start:
                continue

            folds.append({
                "fold_id": i + 1,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            })
        return pd.DataFrame(folds)

class CompositeAlphaBuilder:
    """Xây dựng 4 weight: equal, evidence, ridge, lightgbm."""

    def __init__(self, factor_cols, horizon="fwd_1d"):
        self.factor_cols = factor_cols
        self.horizon = horizon

    def build_daily_ic(self, df: pd.DataFrame) -> pd.DataFrame:
        """IC Spearman per day giữa factor và horizon."""
        in_u = df[df["in_universe"] == 1]
        rows = []
        for date, g in in_u.groupby("timestamp"):
            for factor in self.factor_cols:
                corr, _ = spearmanr(g[factor], g[self.horizon])
                rows.append({
                    "timestamp": date,
                    "factor_id": factor,
                    "horizon": self.horizon,
                    "ic_spearmanr": corr,
                })
        return pd.DataFrame(rows)

    def calculate_ic_period(self, daily_ic: pd.DataFrame,
                            folds: pd.DataFrame) -> pd.DataFrame:
        """Tính ic_mean trên training window của từng fold."""
        ic = daily_ic.copy()
        ic["timestamp"] = pd.to_datetime(ic["timestamp"], utc=True)
        folds = folds.copy()
        folds["train_start"] = pd.to_datetime(folds["train_start"], utc=True)
        folds["train_end"] = pd.to_datetime(folds["train_end"], utc=True)

        for f in self.factor_cols:
            folds[f"ic_mean_{f}"] = None

        for i in folds.index:
            period = ic[
                (ic["timestamp"] >= folds.loc[i, "train_start"]) &
                (ic["timestamp"] <= folds.loc[i, "train_end"]) &
                (ic["horizon"] == self.horizon)
            ]
            means = period.groupby("factor_id")["ic_spearmanr"].mean()
            for f in self.factor_cols:
                folds.loc[i, f"ic_mean_{f}"] = means.get(f, np.nan)

        ic_cols = [f"ic_mean_{f}" for f in self.factor_cols]
        total = folds[ic_cols].sum(axis=1)
        for f in self.factor_cols:
            folds[f"weight_{f}"] = folds[f"ic_mean_{f}"] / total
        return folds

    # ---------- equal / evidence ----------
    def build_composite_alpha(self, df, test_start, test_end,
                             fold_ic, horizon="fwd_1d") -> pd.DataFrame:
        test_start = pd.to_datetime(test_start, utc=True)
        test_end = pd.to_datetime(test_end, utc=True)
        fold_ic = fold_ic.copy()
        fold_ic["test_start"] = pd.to_datetime(fold_ic["test_start"], utc=True)
        fold_ic["test_end"] = pd.to_datetime(fold_ic["test_end"], utc=True)
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        temp = df[(df["timestamp"] >= test_start) & (df["timestamp"] <= test_end)]
        period = temp[["timestamp", "symbol", "in_universe"]
                      + self.factor_cols + [horizon]].copy()

        fold_row = fold_ic[
            (fold_ic["test_start"] == test_start) & (fold_ic["test_end"] == test_end)
        ].iloc[0]
        period["fold_id"] = fold_row["fold_id"]

        evidence_cols = []
        for factor in self.factor_cols:
            w = fold_row[f"weight_{factor}"]
            period[f"evidence_{factor}"] = w * period[factor]
            evidence_cols.append(f"evidence_{factor}")

        period["equal_weight"] = period[self.factor_cols].mean(axis=1)
        period["evidence_weight"] = period[evidence_cols].sum(axis=1)
        return period.drop(columns=evidence_cols)

    def build_all_composite(self, df, fold_ic) -> pd.DataFrame:
        parts = []
        for _, row in fold_ic.iterrows():
            parts.append(self.build_composite_alpha(
                df, row["test_start"], row["test_end"], fold_ic, self.horizon
            ))
        return pd.concat(parts, ignore_index=True)

    # ---------- ridge ----------
    def _ridge_fold(self, df, train_start, train_end, test_start, test_end):
        train = df[
            (df["timestamp"] >= train_start) & (df["timestamp"] <= train_end)
            & (df["in_universe"] == 1)
        ].copy()
        test = df[
            (df["timestamp"] >= test_start) & (df["timestamp"] <= test_end)
            & (df["in_universe"] == 1)
        ].copy()

        x_train = train[self.factor_cols].values
        y_train = train[self.horizon].values
        x_test = test[self.factor_cols].values

        if np.any(np.isnan(x_train)) or np.any(np.isnan(y_train)):
            mask = (~np.isnan(x_train).any(axis=1)) & (~np.isnan(y_train))
            x_train, y_train = x_train[mask], y_train[mask]
        mask_test = ~np.isnan(x_test).any(axis=1)
        x_test = x_test[mask_test]
        test = test.loc[mask_test]

        model = Ridge(alpha=config.RIDGE_ALPHA, random_state=config.RANDOM_SEED)
        model.fit(x_train, y_train)
        out = test[["timestamp", "symbol"]].copy()
        out["weight_ridge"] = model.predict(x_test)
        return out

    def build_ridge(self, df, folds) -> pd.DataFrame:
        parts = []
        for _, row in folds.iterrows():
            p = self._ridge_fold(df, row["train_start"], row["train_end"],
                                 row["test_start"], row["test_end"])
            p["fold_id"] = row["fold_id"]
            parts.append(p)
        return pd.concat(parts, ignore_index=True)

    # ---------- lightgbm ----------
    def _lgbm_fold(self, df, train_start, train_end, test_start, test_end,
                   ratio=None):
        ratio = ratio or config.LGBM_VAL_RATIO
        train = df[
            (df["timestamp"] >= train_start) & (df["timestamp"] <= train_end)
            & (df["in_universe"] == 1)
        ].copy()
        test = df[
            (df["timestamp"] >= test_start) & (df["timestamp"] <= test_end)
            & (df["in_universe"] == 1)
        ].copy()

        n = len(train)
        idx = int((1 - ratio) * n)

        X = train[self.factor_cols].values
        y = train[self.horizon].values
        X_test = test[self.factor_cols].values

        train_ds = lgb.Dataset(X[:idx], label=y[:idx])
        val_ds = lgb.Dataset(X[idx:], label=y[idx:])

        model = lgb.train(
            config.LGBM_PARAMS, train_ds, valid_sets=[val_ds],
            num_boost_round=config.LGBM_PARAMS["n_estimators"],
            callbacks=[lgb.early_stopping(config.LGBM_EARLY_STOPPING)],
        )
        out = test[["timestamp", "symbol"]].copy()
        out["lightgbm_weight"] = model.predict(
            X_test, num_iteration=model.best_iteration
        )
        out["best_iteration"] = model.best_iteration
        return out

    def build_lightgbm(self, df, folds) -> pd.DataFrame:
        parts = []
        for _, row in folds.iterrows():
            p = self._lgbm_fold(df, row["train_start"], row["train_end"],
                                row["test_start"], row["test_end"])
            p["fold_id"] = row["fold_id"]
            parts.append(p)
        return pd.concat(parts, ignore_index=True)

    # ---------- significance test ----------
    @staticmethod
    def _nw_tstat(series, max_lags=4):
        y = series.dropna().values
        n = len(y)
        if n < 2:
            return np.nan, np.nan
        X = np.ones((n, 1))
        m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
        return m.tvalues[0], m.pvalues[0]

    def build_ic_weight_horizon(self, df_weights, weight_col, weight_base):
        """IC per day của weight so với horizon, và delta so với baseline."""
        result = []
        for ts, g in df_weights.groupby("timestamp"):
            a = g[weight_col].values
            b = g[weight_base].values
            y = g[self.horizon].values
            mask = ~np.isnan(a) & ~np.isnan(b) & ~np.isnan(y)
            if mask.sum() < 3:
                continue
            corr, _ = spearmanr(a[mask], y[mask])
            corr_base, _ = spearmanr(b[mask], y[mask])
            result.append({
                "timestamp": ts,
                "ic_spearman": corr,
                "delta_ic": corr - corr_base,
            })
        return pd.DataFrame(result)

    def statistics_sig_ic_weight(self, df_weights, weight_cols,
                                baseline="equal_weight") -> pd.DataFrame:
        rows = []
        for w in weight_cols:
            ih = self.build_ic_weight_horizon(df_weights, w, baseline)
            t_stat, p_val = self._nw_tstat(ih["delta_ic"])
            sig = "1% alpha" if p_val < 0.01 else "not significant"
            rows.append({
                "weight_type": w,
                "ic_mean": ih["ic_spearman"].mean(),
                "delta_ic_mean": ih["delta_ic"].mean(),
                "t_stat": t_stat,
                "p_value": p_val,
                "sig_delta_ic": sig,
            })
        return pd.DataFrame(rows)


class Step3Pipeline:
    """Orchestrator bước 3."""

    def __init__(self, factor_cols, horizon="fwd_1d"):
        self.factor_cols = factor_cols
        self.horizon = horizon
        self.splitter = WalkForwardSplitter()
        self.builder = CompositeAlphaBuilder(factor_cols, horizon)

    def run(self, df: pd.DataFrame) -> dict:
        # Giống notebook 3:
        #   - df_flipped dùng cho IC / equal / evidence
        #   - df_raw     dùng cho ridge / lgbm
        df_raw = df.copy()
        df_flipped = df.copy()
        for f in self.factor_cols:
            df_flipped[f] = -1 * df_flipped[f]

        folds = self.splitter.build()
        ic_daily = self.builder.build_daily_ic(df_flipped)
        fold_ic = self.builder.calculate_ic_period(ic_daily, folds)

        df_ic = self.builder.build_all_composite(df_flipped, fold_ic)
        df_ridge = self.builder.build_ridge(df_raw, folds)
        df_lgbm = self.builder.build_lightgbm(df_raw, folds)

        df_weights = (
            df_ridge.merge(df_lgbm,
                           on=["timestamp", "symbol", "fold_id"],
                           how="inner")
                    .merge(df_ic[["timestamp", "symbol", "fold_id",
                                  "equal_weight", "evidence_weight",
                                  self.horizon]],
                           on=["timestamp", "symbol", "fold_id"],
                           how="inner")
        )

        weight_cols = ["weight_ridge", "lightgbm_weight",
                       "equal_weight", "evidence_weight"]
        stats = self.builder.statistics_sig_ic_weight(
            df_weights, weight_cols, baseline="equal_weight"
        )

        return {
            "df_weights": df_weights,
            "significance": stats,
            "folds": folds,
            "fold_ic": fold_ic,
        }    
    
    
    def save(self, result: dict):
        config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        result["df_weights"].to_parquet(config.WEIGHTS_FILE)
        result["significance"].to_parquet(
            config.DATA_PROCESSED / "weight_significance.parquet"
        )
        logger.info("Saved weights -> %s", config.WEIGHTS_FILE)
"""Bước 1: Load data → Universe → Feature Engineering.

Tương ứng notebook 1 (1_universe_factor_construction.ipynb).

Output:
    - B1_factor_construction.parquet       (in-sample 2020-2021)
    - 2022-2025_factor_construction.parquet (OOT 2022-2025)
"""
import logging
import re

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


class DataLoader:
    """Load raw OHLCV."""

    def __init__(self, path=None):
        self.path = path or config.OHLCV_FILE

    def load(self) -> pd.DataFrame:
        logger.info("Loading OHLCV from %s", self.path)
        df = pd.read_parquet(self.path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df


class UniverseBuilder:
    """Xây dựng universe: loại stablecoin, bad token, tính is_qualified, rank top N."""

    def __init__(self, min_liquidity=None, top_n=None, stablecoins=None):
        self.min_liquidity = min_liquidity or config.MIN_LIQUIDITY
        self.top_n = top_n or config.TOP_N_UNIVERSE
        self.stablecoins = stablecoins or config.STABLECOINS

    @staticmethod
    def is_bad_token(symbol: str) -> bool:
        base = symbol.replace("/USDT", "")
        if any(p in base for p in config.BAD_PATTERNS):
            return True
        if base in config.BAD_BLACKLIST:
            return True
        if re.search(r"\d+$", base):
            return True
        return False

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lọc eligible, bỏ stablecoin, bỏ bad token, tính is_qualified."""
        result = df.copy()
        result = result[result["listing_eligible"] == True]

        base_symbol = result["symbol"].str.replace("/USDT", "", regex=False)
        result["is_stable_coins"] = base_symbol.isin(self.stablecoins)
        result = result[result["is_stable_coins"] == False]

        result["is_qualified"] = np.where(
            result["median_dollar_volume_30d"] >= self.min_liquidity, 1, 0
        )

        result["is_bad"] = result["symbol"].apply(self.is_bad_token)
        result = result[~result["is_bad"]]

        return result.reset_index(drop=True)

    def create_universe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rank theo median_dollar_volume_30d, gán in_rank/in_universe."""
        table = df.copy()
        table["rank_by_liquid"] = table.groupby("timestamp")[
            "median_dollar_volume_30d"
        ].rank(method="dense", ascending=False)

        table["in_rank"] = np.where(table["rank_by_liquid"] <= self.top_n, 1, 0)
        table["in_universe"] = np.where(
            (table["rank_by_liquid"] <= self.top_n) & (table["is_qualified"] == 1),
            1, 0,
        )
        return table.sort_values(
            ["timestamp", "rank_by_liquid"], ascending=[True, True]
        ).reset_index(drop=True)


class FactorBuilder:
    """Tạo momentum, reversal, volatility, liquidity, forward return, z-score."""

    @staticmethod
    def _log_return(df: pd.DataFrame) -> pd.Series:
        return np.log(df["close"]) - np.log(df.groupby("symbol")["close"].shift(1))

    def build_momentum(self, df: pd.DataFrame,
                       windows=None) -> pd.DataFrame:
        windows = windows or config.MOMENTUM_WINDOWS
        table = df.copy()
        table["log_return"] = self._log_return(table)
        for w in windows:
            table[f"momentum_{w}d"] = table.groupby("symbol")["log_return"].transform(
                lambda x, w=w: x.rolling(window=w, min_periods=w).sum()
            )
        return table[["timestamp", "symbol"] + [f"momentum_{w}d" for w in windows]]

    def build_reversal(self, df: pd.DataFrame,
                       windows=None) -> pd.DataFrame:
        windows = windows or config.REVERSAL_WINDOWS
        table = df.copy()
        for w in windows:
            table[f"reversal_{w}d"] = (
                np.log(table["close"])
                - np.log(table.groupby("symbol")["close"].shift(w))
            )
        return table[["timestamp", "symbol"] + [f"reversal_{w}d" for w in windows]]

    def build_volatility(self, df: pd.DataFrame,
                         windows=None) -> pd.DataFrame:
        windows = windows or config.VOL_WINDOWS
        table = df.copy()
        table["log_return"] = self._log_return(table)
        for w in windows:
            table[f"vol_{w}d"] = table.groupby("symbol")["log_return"].transform(
                lambda x, w=w: x.rolling(window=w, min_periods=w).std()
            )
        table["vol_of_vol_14d"] = table.groupby("symbol")["vol_14d"].transform(
            lambda x: x.rolling(window=14, min_periods=14).std()
        )
        return table[["timestamp", "symbol"]
                     + [f"vol_{w}d" for w in windows]
                     + ["vol_of_vol_14d"]]

    def build_liquid(self, df: pd.DataFrame,
                     windows=None) -> pd.DataFrame:
        windows = windows or config.AMIHUD_WINDOWS
        table = df.copy()
        table["log_return"] = self._log_return(table)
        table["amihud_daily"] = abs(table["log_return"]) / table["dollar_volume"]
        for w in windows:
            table[f"amihud_{w}d"] = table.groupby("symbol")["amihud_daily"].transform(
                lambda x, w=w: x.rolling(window=w, min_periods=w).mean()
            )
        return table[["timestamp", "symbol", "log_return"]
                     + [f"amihud_{w}d" for w in windows]]

    def build_forward_return(self, df: pd.DataFrame,
                             windows=None) -> pd.DataFrame:
        windows = windows or config.FORWARD_WINDOWS
        table = df.copy()
        table["log_return"] = self._log_return(table)
        for w in windows:
            table[f"fwd_{w}d"] = table.groupby("symbol")["log_return"].transform(
                lambda x, w=w: x.rolling(window=w, min_periods=w).sum().shift(-w)
            )
        return table[["timestamp", "symbol"] + [f"fwd_{w}d" for w in windows]]

    def build_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge tất cả factor vào df gốc."""
        return (
            df.merge(self.build_momentum(df), on=["timestamp", "symbol"], how="inner")
              .merge(self.build_reversal(df), on=["timestamp", "symbol"], how="inner")
              .merge(self.build_volatility(df), on=["timestamp", "symbol"], how="inner")
              .merge(self.build_liquid(df), on=["timestamp", "symbol"], how="inner")
        )

    # ---------- z-score ----------
    @staticmethod
    def _winsorize(series: pd.Series, lower=None, upper=None) -> pd.Series:
        lower = lower if lower is not None else config.WINSORIZE_LOWER
        upper = upper if upper is not None else config.WINSORIZE_UPPER
        return series.clip(lower=series.quantile(lower), upper=series.quantile(upper))

    @staticmethod
    def _zscore(series: pd.Series):
        std = series.std()
        if std == 0:
            return 0.0
        return (series - series.mean()) / std

    def build_z_score(self, df: pd.DataFrame,
                      factor_cols=None) -> pd.DataFrame:
        """Winsorize + z-score theo timestamp, chỉ áp cho coin trong universe."""
        factor_cols = factor_cols or config.ALL_FACTOR_COLS
        result = df.copy()
        mask = result["in_universe"] == 1

        for factor in factor_cols:
            winsor = (
                result.loc[mask]
                .groupby("timestamp")[factor]
                .transform(lambda x: self._winsorize(x))
            )
            result.loc[mask, f"{factor}_winsorize"] = winsor

            z = (
                result.loc[mask]
                .groupby("timestamp")[f"{factor}_winsorize"]
                .transform(self._zscore)
            )
            result.loc[mask, factor] = z
            result.loc[~mask, factor] = np.nan

            result = result.rename(columns={factor: f"z_{factor}"})
            result = result.drop(columns=[f"{factor}_winsorize"])

        drop_cols = [
            "open", "high", "low", "close", "volume", "dollar_volume",
            "median_dollar_volume_30d", "first_trading_date",
            "listing_age_days", "listing_eligible", "is_qualified",
            "rank_by_liquid", "in_rank",
            "is_stable_coins", "is_bad",
        ]
        return result.drop(columns=[c for c in drop_cols if c in result.columns])


class Step1Pipeline:

    def __init__(self, raw_path=None):
        self.loader = DataLoader(raw_path)
        self.universe = UniverseBuilder()
        self.engineer = FactorBuilder()

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = self.loader.load()
        df = self.universe.build(df)
        df = self.universe.create_universe(df)

        result = self.engineer.build_factors(df)
        result_z = self.engineer.build_z_score(result)

        forward = self.engineer.build_forward_return(df)
        return result_z, forward

    def split_and_save(self, result_z: pd.DataFrame, forward: pd.DataFrame):
        """Chia in-sample (2020-2021) và OOT (2022-2025), lưu parquet."""
        is_mask = (result_z["timestamp"] >= "2020-01-01") & \
                  (result_z["timestamp"] <= config.IS_END)
        oot_mask = (result_z["timestamp"] >= config.DATA_START) & \
                   (result_z["timestamp"] <= config.OOT_END)

        b1 = result_z[is_mask].merge(forward, on=["timestamp", "symbol"], how="left")
        oot = result_z[oot_mask].merge(forward, on=["timestamp", "symbol"], how="left")

        config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        b1.to_parquet(config.B1_FILE)
        oot.to_parquet(config.OOT_FILE)
        logger.info("Saved B1 (%d rows) -> %s", len(b1), config.B1_FILE)
        logger.info("Saved OOT (%d rows) -> %s", len(oot), config.OOT_FILE)
        return b1, oot
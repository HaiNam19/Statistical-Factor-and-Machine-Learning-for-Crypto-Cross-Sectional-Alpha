from pathlib import Path

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_UNIVERSE = DATA_DIR / "universe"
DATA_PROCESSED = DATA_DIR / "processed"

OHLCV_FILE = DATA_UNIVERSE / "ohlcv_daily.parquet"
B1_FILE = DATA_PROCESSED / "B1_factor_construction.parquet"
OOT_FILE = DATA_PROCESSED / "2022-2025_factor_construction.parquet"
WEIGHTS_FILE = DATA_PROCESSED / "D1_df_weightt.parquet"

# ---------- Stablecoins ----------
STABLECOINS = [
    "USDT", "USDC", "DAI", "BUSD", "TUSD", "FDUSD",
    "USDP", "USDD", "UST", "USTC", "FRAX", "RLUSD",
]

# ---------- Universe ----------
MIN_LIQUIDITY = 10_000_000
TOP_N_UNIVERSE = 40
BAD_PATTERNS = ["UP", "DOWN", "BULL", "BEAR", "3L", "3S", "HEDGE"]
BAD_BLACKLIST = ["WBTC", "WETH", "WSTETH", "WBETH", "BTCST", "ETH2"]

# ---------- Factors ----------
MOMENTUM_WINDOWS = [7, 14, 30, 90]
REVERSAL_WINDOWS = [1, 3]
VOL_WINDOWS = [7, 14, 30]
AMIHUD_WINDOWS = [14, 30]
FORWARD_WINDOWS = [1, 5, 14]

ALL_FACTOR_COLS = (
    [f"momentum_{w}d" for w in MOMENTUM_WINDOWS]
    + [f"reversal_{w}d" for w in REVERSAL_WINDOWS]
    + [f"vol_{w}d" for w in VOL_WINDOWS]
    + ["vol_of_vol_14d"]
    + [f"amihud_{w}d" for w in AMIHUD_WINDOWS]
)
ALL_FACTOR_COLS_Z = [f"z_{c}" for c in ALL_FACTOR_COLS]
ALL_FORWARD_COLS = [f"fwd_{w}d" for w in FORWARD_WINDOWS]


ALL_FORWARD_COLS = [f"fwd_{w}d" for w in FORWARD_WINDOWS]

# ---------- Z-score ----------
WINSORIZE_LOWER = 0.05
WINSORIZE_UPPER = 0.95

# ---------- Validation ----------
SIG_LEVELS = ("alpha 1%", "alpha 5%")
MIN_SPREAD_MAGNITUDE = 10
NW_LAG_MAP = {"fwd_1d": 0, "fwd_5d": 4, "fwd_14d": 13}

# ---------- Periods ----------
IN_SAMPLE_START = "2020-09-27"   # notebook 2 filter
IS_END = "2021-12-31"
OOT_START = "2023-01-01"
OOT_END = "2025-12-31"
DATA_START = "2022-01-01"
TRAIN_WINDOW_MONTHS = 12

# ---------- Alpha ----------
RIDGE_ALPHA = 1.0
LGBM_VAL_RATIO = 0.2
LGBM_EARLY_STOPPING = 50
LGBM_PARAMS = {
    "objective": "regression", "metric": "rmse", "boosting_type": "gbdt",
    "num_leaves": 8, "max_depth": 4, "min_data_in_leaf": 20,
    "learning_rate": 0.05, "feature_fraction": 0.8, "bagging_fraction": 0.8,
    "bagging_freq": 5, "reg_alpha": 0.5, "reg_lambda": 0.5, "verbose": -1,
    "n_estimators": 1000, "seed": 42, "deterministic": True,
}

# ---------- Backtest ----------
N_GROUPS = 3
FEE_BPS = 4
SLIPPAGE_BUCKETS = {1: 2, 2: 5, 3: 10, 4: 20}
RANDOM_SEED = 42
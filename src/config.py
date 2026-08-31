
# ============================================================
# CONFIGURATION
# ============================================================

EXCHANGE_ID = "binance"

# ============================================================
# DATA PERIOD
# ============================================================
# Historical period used for:
# - rolling factors
# - liquidity calculation
# - listing age
# - warm-up before research period
#
DATA_START = "2020-01-01"


# ============================================================
# RESEARCH PERIOD
# ============================================================
# Period used for:
# - universe construction
# - factor validation
# - portfolio research
RESEARCH_START = "2023-01-01"
RESEARCH_END = "2025-12-31"


# ============================================================
# UNIVERSE PARAMETERS
# ============================================================

TIMEFRAME = "1d"

MIN_LISTING_DAYS = 180

LIQUIDITY_WINDOW = 30

TOP_N = 40



# ============================================================
# STABLECOINS
# ============================================================

STABLECOINS = {
    "USDT",
    "USDC",
    "BUSD",
    "DAI",
    "TUSD",
    "USDP",
    "FDUSD",
    "USDE",
}


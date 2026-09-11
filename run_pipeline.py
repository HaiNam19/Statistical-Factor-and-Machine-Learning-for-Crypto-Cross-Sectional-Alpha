import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Đảm bảo import src được khi chạy script
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  
from src.step1_feature import Step1Pipeline  
from src.step2_validation import Step2Pipeline  
from src.step3_alpha import Step3Pipeline  
from src.step4_backtest import Step4Pipeline  

logger = logging.getLogger("run_pipeline")


def run_step1():
    logger.info("=" * 60)
    logger.info("STEP 1: Universe & Feature Engineering")
    logger.info("=" * 60)
    pipeline = Step1Pipeline()
    result_z, forward = pipeline.run()
    pipeline.split_and_save(result_z, forward)


def run_step2():
    logger.info("=" * 60)
    logger.info("STEP 2: Statistical Validation")
    logger.info("=" * 60)
    b1 = pd.read_parquet(config.B1_FILE)
    pipeline = Step2Pipeline()
    result = pipeline.run(b1)
    result["eligible"].to_parquet(
        config.DATA_PROCESSED / "eligible_factors.parquet"
    )
    logger.info("Final factors: %s", result["final_factors"])
    return result["final_factors"]


def run_step3(factor_cols):
    logger.info("=" * 60)
    logger.info("STEP 3: Composite Alpha")
    logger.info("=" * 60)
    df = pd.read_parquet(config.OOT_FILE)
    pipeline = Step3Pipeline(factor_cols, horizon="fwd_1d")
    result = pipeline.run(df)
    pipeline.save(result)
    logger.info("Significance:\n%s", result["significance"])


def run_step4(weight_col):
    logger.info("=" * 60)
    logger.info("STEP 4: Backtest")
    logger.info("=" * 60)
    df_weights = pd.read_parquet(config.WEIGHTS_FILE)
    oot = pd.read_parquet(config.OOT_FILE)
    pipeline = Step4Pipeline(weight_col=weight_col)
    result = pipeline.run(df_weights, oot)
    print(result["summary"].to_string(index=False))
    pipeline.plot(result["net"], save_path=config.DATA_PROCESSED / "backtest_plot.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 3, 4],
                       help="Các bước muốn chạy (1-4)")
    parser.add_argument("--weight", type=str, default="equal_weight",
                       help="Loại weight dùng để backtest")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    factor_cols = None
    if 1 in args.steps:
        run_step1()
    if 2 in args.steps:
        factor_cols = run_step2()
    if 3 in args.steps:
        if factor_cols is None:
            eligible = pd.read_parquet(
                config.DATA_PROCESSED / "eligible_factors.parquet"
            )
            factor_cols = eligible["factor_id"].unique().tolist()
        run_step3(factor_cols)
    if 4 in args.steps:
        run_step4(args.weight)


if __name__ == "__main__":
    main()
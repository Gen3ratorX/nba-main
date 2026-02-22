"""
backtest_model.py
Deprecated wrapper. Routes to strict_backtest.py for one canonical report.
"""
import argparse
import sys

from strict_backtest import run_strict_backtest


def main() -> int:
    parser = argparse.ArgumentParser(description='Deprecated: use strict backtest pipeline')
    parser.add_argument('--days', type=int, default=180, help='Lookback days')
    parser.add_argument('--splits', type=int, default=8, help='Rolling split count')
    parser.add_argument(
        '--output',
        type=str,
        default='reports/strict_backtest_latest.json',
        help='Strict report file path',
    )
    args = parser.parse_args()

    print("⚠️ backtest_model.py is deprecated. Running strict canonical backtest...")
    run_strict_backtest(days_back=args.days, n_splits=args.splits, output_path=args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
predictions.py - compatibility wrapper.
Routes legacy CLI calls to the unified pipeline in main.py to prevent drift.
"""
import argparse
import sys

from main import UnifiedNBAPredictionSystem
from nbautils import log_warning, log_error


def main():
    parser = argparse.ArgumentParser(
        description='Legacy predictions entrypoint (uses unified pipeline)'
    )
    parser.add_argument('--bankroll', type=float, default=1000, help='Bankroll amount')
    parser.add_argument('--days-back', type=int, default=120, help='Days back for data pull')
    parser.add_argument('--days-ahead', type=int, default=7, help='Days ahead for schedule pull')
    parser.add_argument('--max-bets', type=int, default=3, help='Maximum bet recommendations')
    parser.add_argument('--no-update', action='store_true', help='Skip data download')
    parser.add_argument('--no-retrain', action='store_true', help='Use latest saved model instead of retraining')

    # Backward-compatible legacy flags (now ignored by canonical pipeline).
    parser.add_argument('--moneyline', action='store_true', help='Deprecated')
    parser.add_argument('--totals', action='store_true', help='Deprecated')
    parser.add_argument('--tomorrow', action='store_true', help='Deprecated')
    parser.add_argument('--date', type=str, default=None, help='Deprecated')
    parser.add_argument('--min-edge', type=float, default=None, help='Deprecated')
    parser.add_argument('--typical-total', type=float, default=None, help='Deprecated')
    parser.add_argument('--show-all', action='store_true', help='Deprecated')

    args = parser.parse_args()

    if any([
        args.moneyline,
        args.totals,
        args.tomorrow,
        args.date,
        args.min_edge is not None,
        args.typical_total is not None,
        args.show_all,
    ]):
        log_warning(
            "Legacy flags detected; running unified pipeline defaults for consistency. "
            "Use main.py for canonical options."
        )

    system = UnifiedNBAPredictionSystem(bankroll=args.bankroll)

    try:
        success = system.full_pipeline(
            retrain_model=not args.no_retrain,
            update_data=not args.no_update,
            days_back=args.days_back,
            days_ahead=args.days_ahead,
            max_bets=args.max_bets,
        )
        return 0 if success else 1
    except KeyboardInterrupt:
        return 1
    except Exception as exc:
        log_error(f"Predictions wrapper failed: {exc}")
        return 1


if __name__ == '__main__':
    sys.exit(main())

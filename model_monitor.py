"""
model_monitor.py - Track model performance and alert on drift.

Monitors rolling accuracy, Brier score, and calibration.
Sends Telegram alert if performance drops below thresholds.

Designed to run after each pipeline execution.

Usage:
    python model_monitor.py                  # check & alert
    python model_monitor.py --history        # show rolling history
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

# Thresholds for alerts
ACCURACY_ALERT = 0.52        # below 52% over rolling window = alert
BRIER_ALERT = 0.26           # above 0.26 = alert
COLD_STREAK_ALERT = 6        # 6+ consecutive losses = alert
ROLLING_WINDOW = 30          # evaluate over last 30 predictions

MONITOR_FILE = Path("data/model_monitor_history.json")


def _load_predictions() -> list[dict]:
    """Load prediction history from tracker."""
    tracker_path = Path("data/prediction_history.json")
    if not tracker_path.exists():
        return []
    try:
        with open(tracker_path) as f:
            data = json.load(f)
        # Handle both list format and dict with 'predictions' key
        if isinstance(data, list):
            return data
        return data.get("predictions", [])
    except Exception:
        return []


def _load_monitor_history() -> list[dict]:
    """Load historical monitoring snapshots."""
    if not MONITOR_FILE.exists():
        return []
    try:
        with open(MONITOR_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def _save_monitor_history(history: list[dict]):
    MONITOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MONITOR_FILE, "w") as f:
        json.dump(history[-365:], f, indent=2)  # keep last year


def check_performance() -> dict:
    """Evaluate current model performance and detect drift."""
    predictions = _load_predictions()

    # Filter to resolved predictions only
    resolved = [
        p for p in predictions
        if p.get("result") in ("win", "loss", "W", "L", True, False, 1, 0)
    ]

    if len(resolved) < 5:
        return {
            "status": "insufficient_data",
            "message": f"Only {len(resolved)} resolved predictions. Need at least 5.",
            "alerts": [],
        }

    # Normalize results
    for p in resolved:
        r = p.get("result")
        if r in ("win", "W", True, 1):
            p["_won"] = True
        else:
            p["_won"] = False

    # Rolling window
    recent = resolved[-ROLLING_WINDOW:]
    wins = sum(1 for p in recent if p["_won"])
    total = len(recent)
    accuracy = wins / total

    # Brier score (if probabilities available)
    brier = None
    probs_available = all("probability" in p or "win_probability" in p for p in recent)
    if probs_available:
        brier_sum = 0
        for p in recent:
            prob = p.get("probability") or p.get("win_probability", 0.5)
            if isinstance(prob, str):
                prob = float(prob.strip("%")) / 100
            outcome = 1.0 if p["_won"] else 0.0
            brier_sum += (prob - outcome) ** 2
        brier = brier_sum / total

    # Cold streak detection
    streak = 0
    for p in reversed(resolved):
        if not p["_won"]:
            streak += 1
        else:
            break

    # Generate alerts
    alerts = []

    if accuracy < ACCURACY_ALERT:
        alerts.append({
            "type": "low_accuracy",
            "severity": "high",
            "message": f"Rolling accuracy dropped to {accuracy:.1%} "
                       f"({wins}/{total} over last {ROLLING_WINDOW} picks). "
                       f"Threshold: {ACCURACY_ALERT:.0%}",
        })

    if brier is not None and brier > BRIER_ALERT:
        alerts.append({
            "type": "high_brier",
            "severity": "medium",
            "message": f"Brier score at {brier:.4f} (threshold: {BRIER_ALERT}). "
                       f"Probability estimates are miscalibrated.",
        })

    if streak >= COLD_STREAK_ALERT:
        alerts.append({
            "type": "cold_streak",
            "severity": "high",
            "message": f"Current losing streak: {streak} consecutive losses. "
                       f"Consider pausing bets until model recovers.",
        })

    # Overall status
    if alerts:
        status = "alert"
    elif accuracy > 0.60:
        status = "excellent"
    elif accuracy > 0.55:
        status = "good"
    else:
        status = "ok"

    report = {
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "rolling_window": ROLLING_WINDOW,
        "total_resolved": len(resolved),
        "recent_count": total,
        "accuracy": round(accuracy, 4),
        "wins": wins,
        "losses": total - wins,
        "brier_score": round(brier, 4) if brier else None,
        "current_streak": -streak if streak > 0 else 0,
        "alerts": alerts,
    }

    # Save snapshot
    history = _load_monitor_history()
    history.append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "accuracy": report["accuracy"],
        "brier": report["brier_score"],
        "total": report["total_resolved"],
    })
    _save_monitor_history(history)

    return report


def format_alert_message(report: dict) -> str | None:
    """Format Telegram alert message. Returns None if no alerts."""
    alerts = report.get("alerts", [])
    if not alerts:
        return None

    lines = ["⚠️ *NBA MODEL ALERT*\n"]

    for alert in alerts:
        severity_icon = "🔴" if alert["severity"] == "high" else "🟡"
        lines.append(f"{severity_icon} {alert['message']}")

    lines.append(f"\n📊 Rolling stats (last {report.get('rolling_window', 20)}):")
    lines.append(f"  Accuracy: {report.get('accuracy', 0):.1%} ({report.get('wins', 0)}W-{report.get('losses', 0)}L)")
    if report.get("brier_score"):
        lines.append(f"  Brier: {report['brier_score']:.4f}")
    lines.append(f"  Total predictions: {report.get('total_resolved', 0)}")

    return "\n".join(lines)


def format_status_message(report: dict) -> str:
    """Format a brief status line (non-alert, for daily summary)."""
    if report.get("status") == "insufficient_data":
        return f"⚪ Model: {report.get('message', 'Not enough data yet')}"
    icons = {"excellent": "🟢", "good": "🟢", "ok": "🟡", "alert": "🔴"}
    icon = icons.get(report["status"], "⚪")
    acc = report.get("accuracy", 0)
    msg = f"{icon} Model: {acc:.1%} accuracy ({report.get('wins', 0)}W-{report.get('losses', 0)}L last {report.get('recent_count', 0)})"
    if report.get("brier_score"):
        msg += f" | Brier: {report['brier_score']:.4f}"
    return msg


def send_alert_if_needed(report: dict) -> bool:
    """Send Telegram alert if performance has degraded."""
    message = format_alert_message(report)
    if not message:
        return False

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("⚠️  Telegram credentials not set — alert not sent")
        print(message)
        return False

    from notify_telegram import send_telegram
    return send_telegram(message, token, chat_id)


def show_history():
    """Print rolling performance history."""
    history = _load_monitor_history()
    if not history:
        print("No monitoring history yet.")
        return

    print(f"\n{'Date':<12} {'Accuracy':<10} {'Brier':<10} {'Total'}")
    print("-" * 45)
    for h in history[-30:]:
        brier = f"{h['brier']:.4f}" if h.get("brier") else "N/A"
        print(f"{h['date']:<12} {h['accuracy']:.1%}      {brier:<10} {h['total']}")


def main():
    parser = argparse.ArgumentParser(description="NBA Model Monitor")
    parser.add_argument("--history", action="store_true", help="Show performance history")
    parser.add_argument("--alert", action="store_true", help="Send alert if needed")
    args = parser.parse_args()

    if args.history:
        show_history()
        return

    report = check_performance()
    print(format_status_message(report))

    if report["alerts"]:
        print(f"\n⚠️  {len(report['alerts'])} alert(s) detected!")
        for alert in report["alerts"]:
            print(f"  - [{alert['severity'].upper()}] {alert['message']}")

        if args.alert:
            send_alert_if_needed(report)
    else:
        print("✅ No alerts — model performing within thresholds")


if __name__ == "__main__":
    main()

"""
notify_telegram.py - Send daily NBA picks to Telegram.

Parses the pipeline log file for picks and sends a formatted message.
Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.

Usage:
    python notify_telegram.py logs/pipeline_20260314.log
    python notify_telegram.py  # reads latest log in logs/
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib import request, parse
import json


def _find_latest_log() -> Path | None:
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return None
    logs = sorted(logs_dir.glob("pipeline_*.log"), reverse=True)
    return logs[0] if logs else None


def _parse_picks(log_text: str) -> dict:
    """Extract picks and game slate from pipeline log output."""
    result = {
        "top_bets": [],
        "full_slate": [],
        "no_bets": False,
        "skipped_no_odds": 0,
        "skipped_calibration": 0,
    }

    # Parse top recommendations
    in_top = False
    current_bet = {}
    for line in log_text.splitlines():
        stripped = line.strip()

        # Detect "NO QUALIFYING BETS"
        if "NO QUALIFYING BETS" in line:
            result["no_bets"] = True

        # Count skipped
        if "GAMES SKIPPED (No Vegas Odds" in line:
            # Count lines after until next section
            pass
        m = re.search(r"Total: (\d+) games", line)
        if m and "No Vegas" in log_text[max(0, log_text.index(line) - 200):log_text.index(line)]:
            result["skipped_no_odds"] = int(m.group(1))

        # Top recommendations section
        if "TOP RECOMMENDATIONS" in line:
            in_top = True
            continue
        if in_top and line.startswith("#") or (in_top and re.match(r"\s*#\d+", line)):
            if current_bet:
                result["top_bets"].append(current_bet)
            current_bet = {}
            # Extract date
            dm = re.search(r"📅\s*(.+)", line)
            if dm:
                current_bet["date"] = dm.group(1).strip()

        if in_top:
            # Game matchup
            gm = re.search(r"(.+?)\s+@\s+(.+)", stripped)
            if gm and "@" in stripped and "MONEYLINE" not in stripped:
                current_bet["away"] = gm.group(1).strip()
                current_bet["home"] = gm.group(2).strip()

            # Moneyline pick
            mm = re.search(r"MONEYLINE:\s*(.+)", stripped)
            if mm:
                current_bet["type"] = "ML"
                current_bet["pick"] = mm.group(1).strip()

            # Totals pick
            tm = re.search(r"TOTAL:\s*(.+)", stripped)
            if tm:
                current_bet["type"] = "TOTAL"
                current_bet["pick"] = tm.group(1).strip()

            # Win probability
            wp = re.search(r"Win Probability:\s*([\d.]+)%", stripped)
            if wp:
                current_bet["prob"] = wp.group(1)

            # Bet amount
            ba = re.search(r"BET:\s*\$([\d,.]+)", stripped)
            if ba:
                current_bet["amount"] = ba.group(1)

            # Edge
            em = re.search(r"Edge:\s*([\d.]+)%", stripped)
            if em:
                current_bet["edge"] = em.group(1)

            # Confidence
            cm = re.search(r"Confidence:\s*(\w+)", stripped)
            if cm:
                current_bet["confidence"] = cm.group(1)

            # End of top recommendations
            if "Total Exposure" in line or "FULL SLATE" in line:
                if current_bet:
                    result["top_bets"].append(current_bet)
                    current_bet = {}
                in_top = False

        # Parse full slate table
        if re.match(r"\d{4}-\d{2}-\d{2}\s+\|", stripped):
            parts = [p.strip() for p in stripped.split("|")]
            if len(parts) >= 7:
                result["full_slate"].append({
                    "date": parts[0],
                    "home": parts[1],
                    "away": parts[2],
                    "pick": parts[3],
                    "prob": parts[4],
                })

    return result


def _format_telegram_message(picks: dict) -> str:
    """Format picks into a Telegram message with emojis."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"🏀 *NBA PICKS — {now}*\n"]

    if picks["no_bets"]:
        lines.append("🚫 No qualifying bets today.")
        lines.append("_All games failed minimum edge requirements._")
        lines.append("_This means the system is being selective — good._")
    elif picks["top_bets"]:
        lines.append(f"💎 *TOP {len(picks['top_bets'])} BETS:*\n")
        for i, bet in enumerate(picks["top_bets"], 1):
            matchup = f"{bet.get('away', '?')} @ {bet.get('home', '?')}"
            pick = bet.get("pick", "?")
            bet_type = bet.get("type", "ML")
            prob = bet.get("prob", "?")
            edge = bet.get("edge", "?")
            amount = bet.get("amount", "?")
            conf = bet.get("confidence", "?")

            if bet_type == "ML":
                lines.append(f"*{i}. {matchup}*")
                lines.append(f"   🏆 {pick} (ML)")
                lines.append(f"   📊 {prob}% win prob | Edge: {edge}%")
                lines.append(f"   💰 ${amount} | Confidence: {conf}")
            else:
                lines.append(f"*{i}. {matchup}*")
                lines.append(f"   📊 {pick}")
                lines.append(f"   💰 ${amount} | Edge: {edge}%")
            lines.append("")

    # Full slate summary
    if picks["full_slate"]:
        lines.append(f"📋 *FULL SLATE ({len(picks['full_slate'])} games):*")
        for game in picks["full_slate"]:
            prob = game["prob"].replace("%", "")
            lines.append(f"  {game['away']} @ {game['home']} → {game['pick']} ({game['prob']})")
        lines.append("")

    if picks["skipped_no_odds"]:
        lines.append(f"⚠️ {picks['skipped_no_odds']} games skipped (no odds)")

    lines.append("_Generated by NBA Prediction System v2.0_")
    return "\n".join(lines)


def send_telegram(message: str, token: str, chat_id: str) -> bool:
    """Send message via Telegram Bot API (stdlib only, no requests needed)."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = parse.urlencode({
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": "true",
    }).encode("utf-8")

    try:
        req = request.Request(url, data=data)
        with request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
            if result.get("ok"):
                print(f"✅ Telegram message sent to chat {chat_id}")
                return True
            else:
                print(f"❌ Telegram API error: {result}")
                return False
    except Exception as e:
        print(f"❌ Failed to send Telegram message: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Send NBA picks to Telegram")
    parser.add_argument("log_file", nargs="?", help="Path to pipeline log file")
    args = parser.parse_args()

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("❌ Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables")
        print("\nTo set up:")
        print("1. Message @BotFather on Telegram → /newbot → copy the token")
        print("2. Message your bot, then visit:")
        print("   https://api.telegram.org/bot<TOKEN>/getUpdates")
        print("   Find your chat_id in the response")
        print("3. Add both as GitHub repo secrets:")
        print("   Settings → Secrets → New repository secret")
        sys.exit(1)

    # Find log file
    log_path = Path(args.log_file) if args.log_file else _find_latest_log()
    if not log_path or not log_path.exists():
        print("❌ No log file found. Run the pipeline first.")
        sys.exit(1)

    log_text = log_path.read_text()
    picks = _parse_picks(log_text)
    message = _format_telegram_message(picks)

    print(f"\n--- Message Preview ---\n{message}\n---\n")
    send_telegram(message, token, chat_id)


if __name__ == "__main__":
    main()

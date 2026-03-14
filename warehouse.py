"""
warehouse.py

SQLite warehouse for centralized, append-friendly storage of:
- canonical games (historical/current/upcoming)
- odds snapshots (live pulls)
- feature/prediction/decision snapshots per run

This is designed to be safe-by-default:
- keeps raw JSON blobs where useful
- never deletes; uses upserts for canonical games
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


def _utcnow_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _game_id(game_date: str, home_team: str, away_team: str) -> str:
    # Stable identifier for a scheduled matchup in a given season day.
    return f"{game_date}_{home_team}_{away_team}"


@dataclass(frozen=True)
class WarehouseConfig:
    path: Path = Path(os.getenv("WAREHOUSE_DB_PATH", "data/warehouse.sqlite")).resolve()


class NBAWarehouse:
    def __init__(self, config: WarehouseConfig | None = None):
        self.config = config or WarehouseConfig()
        _ensure_parent_dir(self.config.path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.config.path))
        conn.row_factory = sqlite3.Row
        # Basic durability; still fast enough for local runs.
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                  run_id TEXT PRIMARY KEY,
                  created_at TEXT NOT NULL,
                  meta_json TEXT
                );

                CREATE TABLE IF NOT EXISTS games (
                  game_id TEXT PRIMARY KEY,
                  game_date TEXT NOT NULL,
                  home_team TEXT NOT NULL,
                  away_team TEXT NOT NULL,
                  season TEXT,
                  source TEXT,
                  home_score INTEGER,
                  away_score INTEGER,
                  updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date);

                CREATE TABLE IF NOT EXISTS odds_snapshots (
                  odds_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  run_id TEXT,
                  fetched_at TEXT NOT NULL,
                  bookmaker TEXT,
                  game_date TEXT,
                  home_team TEXT,
                  away_team TEXT,
                  ml_home INTEGER,
                  ml_away INTEGER,
                  spread REAL,
                  spread_price INTEGER,
                  total_line REAL,
                  total_price INTEGER,
                  raw_json TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_odds_date ON odds_snapshots(game_date);
                CREATE INDEX IF NOT EXISTS idx_odds_run ON odds_snapshots(run_id);

                CREATE TABLE IF NOT EXISTS predictions (
                  pred_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  run_id TEXT,
                  game_id TEXT,
                  created_at TEXT NOT NULL,
                  p_home REAL,
                  p_away REAL,
                  totals_pred REAL,
                  totals_unc REAL,
                  totals_lower REAL,
                  totals_upper REAL,
                  model_meta_json TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_preds_game ON predictions(game_id);
                CREATE INDEX IF NOT EXISTS idx_preds_run ON predictions(run_id);

                CREATE TABLE IF NOT EXISTS decisions (
                  dec_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  run_id TEXT,
                  game_id TEXT,
                  created_at TEXT NOT NULL,
                  bet_type TEXT,
                  side TEXT,
                  line REAL,
                  odds INTEGER,
                  stake REAL,
                  edge REAL,
                  ev REAL,
                  reason TEXT,
                  meta_json TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_decs_game ON decisions(game_id);
                CREATE INDEX IF NOT EXISTS idx_decs_run ON decisions(run_id);
                """
            )

    def start_run(self, meta: Optional[Dict[str, Any]] = None) -> str:
        run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs(run_id, created_at, meta_json) VALUES(?,?,?)",
                (run_id, _utcnow_iso(), json.dumps(meta or {}, sort_keys=True)),
            )
        return run_id

    def upsert_games_from_df(self, df: pd.DataFrame, source: str, season: str | None = None) -> int:
        """
        Accepts a dataframe with at least:
        - game_date, home_team, away_team
        Optional: home_score, away_score
        """
        if df is None or df.empty:
            return 0

        cols = set(df.columns)
        required = {"game_date", "home_team", "away_team"}
        if not required.issubset(cols):
            return 0

        rows = []
        for _, r in df.iterrows():
            game_date = str(r["game_date"])[:10]
            home_team = str(r["home_team"])
            away_team = str(r["away_team"])
            gid = _game_id(game_date, home_team, away_team)
            rows.append(
                (
                    gid,
                    game_date,
                    home_team,
                    away_team,
                    season,
                    source,
                    int(r["home_score"]) if "home_score" in cols and pd.notna(r["home_score"]) else None,
                    int(r["away_score"]) if "away_score" in cols and pd.notna(r["away_score"]) else None,
                    _utcnow_iso(),
                )
            )

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO games(
                  game_id, game_date, home_team, away_team, season, source,
                  home_score, away_score, updated_at
                )
                VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(game_id) DO UPDATE SET
                  home_score=COALESCE(excluded.home_score, games.home_score),
                  away_score=COALESCE(excluded.away_score, games.away_score),
                  season=COALESCE(excluded.season, games.season),
                  source=COALESCE(excluded.source, games.source),
                  updated_at=excluded.updated_at
                """,
                rows,
            )

        return len(rows)

    def ingest_odds_snapshot_df(self, run_id: str, odds_df: pd.DataFrame, raw_json: Optional[str] = None) -> int:
        """
        Stores the odds dataframe that comes from VegasOddsFetcher (FanDuel etc).
        Expected columns (if present): game_date, home_team, away_team, ml_home, ml_away, total_line, spread, spread_price
        """
        if odds_df is None or odds_df.empty:
            return 0

        cols = set(odds_df.columns)
        fetched_at = _utcnow_iso()
        bookmaker = None
        if "bookmaker" in cols and odds_df["bookmaker"].notna().any():
            bookmaker = str(odds_df["bookmaker"].dropna().iloc[0])

        rows = []
        for _, r in odds_df.iterrows():
            game_date = str(r["game_date"])[:10] if "game_date" in cols else None
            rows.append(
                (
                    run_id,
                    fetched_at,
                    bookmaker,
                    game_date,
                    str(r["home_team"]) if "home_team" in cols else None,
                    str(r["away_team"]) if "away_team" in cols else None,
                    int(r["ml_home"]) if "ml_home" in cols and pd.notna(r["ml_home"]) else None,
                    int(r["ml_away"]) if "ml_away" in cols and pd.notna(r["ml_away"]) else None,
                    float(r["spread"]) if "spread" in cols and pd.notna(r["spread"]) else None,
                    int(r["spread_price"]) if "spread_price" in cols and pd.notna(r["spread_price"]) else None,
                    float(r["total_line"]) if "total_line" in cols and pd.notna(r["total_line"]) else None,
                    int(r["total_price"]) if "total_price" in cols and pd.notna(r["total_price"]) else None,
                    raw_json,
                )
            )

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO odds_snapshots(
                  run_id, fetched_at, bookmaker, game_date, home_team, away_team,
                  ml_home, ml_away, spread, spread_price, total_line, total_price, raw_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
        return len(rows)

    def ingest_predictions_df(self, run_id: str, preds_df: pd.DataFrame, model_meta: Optional[Dict[str, Any]] = None) -> int:
        if preds_df is None or preds_df.empty:
            return 0

        cols = set(preds_df.columns)
        required = {"game_date", "home_team", "away_team", "home_win_probability", "away_win_probability"}
        if not required.issubset(cols):
            return 0

        created_at = _utcnow_iso()
        meta_json = json.dumps(model_meta or {}, sort_keys=True)

        rows = []
        for _, r in preds_df.iterrows():
            game_date = str(r["game_date"])[:10]
            home_team = str(r["home_team"])
            away_team = str(r["away_team"])
            gid = _game_id(game_date, home_team, away_team)

            totals = r.get("totals_pred")
            totals_pred = totals.get("predicted_total") if isinstance(totals, dict) else None
            totals_unc = totals.get("uncertainty") if isinstance(totals, dict) else None
            totals_lower = totals.get("lower") if isinstance(totals, dict) else None
            totals_upper = totals.get("upper") if isinstance(totals, dict) else None

            rows.append(
                (
                    run_id,
                    gid,
                    created_at,
                    float(r["home_win_probability"]),
                    float(r["away_win_probability"]),
                    float(totals_pred) if totals_pred is not None else None,
                    float(totals_unc) if totals_unc is not None else None,
                    float(totals_lower) if totals_lower is not None else None,
                    float(totals_upper) if totals_upper is not None else None,
                    meta_json,
                )
            )

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO predictions(
                  run_id, game_id, created_at, p_home, p_away, totals_pred, totals_unc,
                  totals_lower, totals_upper, model_meta_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
        return len(rows)

    def ingest_decisions(self, run_id: str, decisions: Iterable[Dict[str, Any]]) -> int:
        rows = []
        created_at = _utcnow_iso()
        for d in decisions or []:
            game = d.get("game_info") or {}
            game_date = str(game.get("date", ""))[:10]
            home = str(game.get("home", ""))
            away = str(game.get("away", ""))
            if not game_date or not home or not away:
                continue
            gid = _game_id(game_date, home, away)
            rows.append(
                (
                    run_id,
                    gid,
                    created_at,
                    d.get("bet_type"),
                    d.get("direction") or d.get("side"),
                    float(d["vegas_line"]) if "vegas_line" in d and d["vegas_line"] is not None else None,
                    int(d["odds"]) if "odds" in d and d["odds"] is not None else None,
                    float(d["bet_amount"]) if "bet_amount" in d and d["bet_amount"] is not None else None,
                    float(d["edge"]) if "edge" in d and d["edge"] is not None else None,
                    float(d["expected_value"]) if "expected_value" in d and d["expected_value"] is not None else None,
                    d.get("reason"),
                    json.dumps(d, sort_keys=True, default=str),
                )
            )

        if not rows:
            return 0

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO decisions(
                  run_id, game_id, created_at, bet_type, side, line, odds, stake, edge, ev, reason, meta_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
        return len(rows)


    def export_odds_to_csv(self, csv_path: str = "data/odds_history/odds_history.csv") -> int:
        """
        Export all odds snapshots from the warehouse into the CSV history file,
        merging with any existing CSV data. This ensures the training pipeline
        can pick up odds even if the CSV archival missed some runs.
        """
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT game_date, fetched_at, bookmaker,
                       home_team, away_team,
                       ml_home, ml_away, spread, spread_price,
                       total_line, total_price
                FROM odds_snapshots
                WHERE ml_home IS NOT NULL AND ml_away IS NOT NULL
                """
            ).fetchall()

        if not rows:
            return 0

        wh_df = pd.DataFrame(
            rows,
            columns=[
                "game_date", "fetched_at", "sportsbook",
                "home_team", "away_team",
                "ml_home", "ml_away", "spread_home", "spread_home_odds",
                "total_line", "over_odds",
            ],
        )
        # Fill columns the CSV expects but warehouse doesn't store
        for col in ["game_time", "spread_away", "spread_away_odds", "under_odds"]:
            if col not in wh_df.columns:
                wh_df[col] = pd.NA

        if csv_path.exists():
            try:
                existing = pd.read_csv(csv_path)
                merged = pd.concat([existing, wh_df], ignore_index=True)
            except Exception:
                merged = wh_df
        else:
            merged = wh_df

        merged = merged.drop_duplicates(
            subset=["game_date", "home_team", "away_team"],
            keep="last",
        )
        merged.to_csv(csv_path, index=False)
        return len(merged)


def sync_canonical_csvs_to_warehouse(wh: NBAWarehouse) -> Dict[str, int]:
    """
    Convenience: ingest the canonical CSVs into the warehouse for centralized queries.
    """
    out: Dict[str, int] = {}
    hist = Path("data/historical/historical_games_balldontlie.csv")
    cur = Path("data/current/current_season_espn.csv")
    upc = Path("data/upcoming/upcoming_games.csv")

    if hist.exists():
        out["historical"] = wh.upsert_games_from_df(pd.read_csv(hist), source="balldontlie")
    else:
        out["historical"] = 0

    if cur.exists():
        out["current"] = wh.upsert_games_from_df(pd.read_csv(cur), source="espn")
    else:
        out["current"] = 0

    if upc.exists():
        out["upcoming"] = wh.upsert_games_from_df(pd.read_csv(upc), source="espn_upcoming")
    else:
        out["upcoming"] = 0

    return out


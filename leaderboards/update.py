"""Daily update: the top-20 leaders, the pool probe for ranks up to TOP_N, snapshots, stats.

    python -m leaderboards.update              # today's UTC date
    python -m leaderboards.update --force      # overwrite today's daily snapshot
    python -m leaderboards.update --date 2026-11-20

Outcomes (see PLAN.md):
- Login expired or the leaderboard reply is missing: exit 1, nothing written.
- Pool probe works: coverage "pool". Writes the daily snapshot, players, records and pool.
- Pool probe fails: coverage "top20". Writes only today.json (the top 20, with a warning),
  leaves players, records and the pool unchanged, and exits 0. The job never silently drops
  ranks 21-TOP_N, so the stats do not record a false mass drop-out.
Reruns for the same date are idempotent: an existing pool-coverage daily file is reused
unless --force.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import pool as pool_mod
from . import stats
from .config import BOARD, LEADER_ROWS, MAX_NEW_CANDIDATES, MAX_PROBES, SEED_MIN_RATING, TOP_N

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


class DataError(RuntimeError):
    pass


def _utc_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _read(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def validate_leaders(rows: list[dict]) -> list[dict]:
    """The server's top list must be non-empty with ranks 1..k in order."""
    if not rows:
        raise DataError("the leaderboard reply had no rows; nothing written")
    for i, r in enumerate(rows, 1):
        if r.get("rank") != i or not r.get("uid") or not r.get("username") \
                or not isinstance(r.get("rating"), int):
            raise DataError(f"bad leader row {i}: {r!r}; nothing written")
    return rows


def validate_board(rows: list[dict], top_n: int) -> list[dict]:
    for r in rows:
        if not isinstance(r.get("rank"), int) or not (1 <= r["rank"] <= top_n) \
                or not r.get("uid") or not r.get("username") or not isinstance(r.get("rating"), int):
            raise DataError(f"bad board row: {r!r}")
    return rows


def _previous_date(dates: list[str], date: str):
    return max((d for d in dates if d < date), default=None)


def run(data_dir: Path, client, date: str | None = None, force: bool = False, top_n: int = TOP_N,
        board: str = BOARD, now: str | None = None, seed_min: int = SEED_MIN_RATING,
        max_probes: int = MAX_PROBES, max_new: int = MAX_NEW_CANDIDATES) -> dict:
    """One daily update. `client` provides leaders(), watch(), scraper(), probe(cands)."""
    date = date or _utc_today()
    now = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    warnings: list[str] = []

    leaders = validate_leaders(client.leaders())            # raises: nothing written
    leader_rows = [{"rank": r["rank"], "uid": r["uid"], "username": r["username"], "rating": r["rating"]}
                   for r in leaders]

    watch = []
    try:
        watch = client.watch()
    except Exception as e:   # the watch list only adds candidates; the job goes on without it
        warnings.append(f"watch list unavailable ({type(e).__name__}); pool grows from other sources")

    pool_path = data_dir / "pool.json"
    pool = _read(pool_path, {})
    players = _read(data_dir / "players.json", {})
    on_board_prev = [{"uid": uid, "username": rec["username"]}
                     for uid, rec in players.items() if rec.get("currently_on_board")]
    scraper = client.scraper()

    cands = pool_mod.gather_candidates(pool, leader_rows, on_board_prev, watch, scraper, date,
                                       seed_min=seed_min, max_new=max_new)[:max_probes]
    try:
        results, failures = client.probe(cands)
    except Exception as e:
        return _top20_only(data_dir, leader_rows, date, now, board, top_n,
                           warnings + [f"pool probe failed ({type(e).__name__}: {e}); "
                                       f"publishing the top {len(leader_rows)} only"],
                           probes=0)

    rows = validate_board(pool_mod.build_board(results, leader_rows, top_n), top_n)
    pool = pool_mod.apply_probes(pool, results, date)
    if len(rows) < top_n:
        warnings.append(f"pool covers {len(rows)} of the top {top_n} ranks; players we have not seen "
                        f"may be missing from ranks {len(leader_rows) + 1}-{top_n}")
    if failures:
        warnings.append(f"{len(failures)} profile probes timed out today")

    daily_path = data_dir / "daily" / f"{date}.json"
    if daily_path.exists() and not force:
        snap = _read(daily_path, None)
        status = "kept existing snapshot"
    else:
        snap = {"date": date, "fetched_at": now, "board": board, "top_n": top_n, "coverage": "pool",
                "count": len(rows), "players": rows}
        _write(daily_path, snap)
        status = "wrote snapshot"

    index_path = data_dir / "index.json"
    old_dates = _read(index_path, {"dates": []}).get("dates", [])
    prev_date = _previous_date(old_dates, date)
    dates = sorted(set(old_dates) | {date})
    latest = dates[-1]

    players, info = stats.apply_snapshot(players, snap, date, prev_date)
    records = stats.build_records(players, date, top_n=top_n, movers=info["movers"])

    today = {**snap, "coverage": "pool", "rows_expected": top_n, "warnings": warnings,
             "stats_updated": True, "probes_sent": len(cands), "probes_failed": len(failures),
             "entered": info["entered"], "left": info["left"]}
    if date == latest:   # today.json is the newest snapshot only (backfills do not replace it)
        _write(data_dir / "today.json", today)
    _write(index_path, {"dates": dates, "latest": latest})
    _write(data_dir / "players.json", players)
    _write(data_dir / "records.json", records)
    _write(pool_path, pool)

    return {"status": status, "coverage": "pool", "date": date, "count": len(rows),
            "probes_sent": len(cands), "pool_members": len(pool_mod.pool_members(pool)),
            "prev_date": prev_date, "warnings": warnings,
            "entered": [e["username"] for e in info["entered"]],
            "left": [e["username"] for e in info["left"]]}


def _top20_only(data_dir: Path, leader_rows: list[dict], date: str, now: str, board: str,
                top_n: int, warnings: list[str], probes: int) -> dict:
    """Publish the server's top list only. Players, records and pool are left as they were."""
    today = {"date": date, "fetched_at": now, "board": board, "top_n": top_n, "coverage": "top20",
             "rows_expected": top_n, "count": len(leader_rows), "players": leader_rows,
             "warnings": warnings, "stats_updated": False, "probes_sent": probes,
             "probes_failed": 0, "entered": [], "left": []}
    _write(data_dir / "today.json", today)
    return {"status": "top20 only", "coverage": "top20", "date": date, "count": len(leader_rows),
            "probes_sent": probes, "warnings": warnings, "entered": [], "left": []}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="overwrite the daily snapshot for this date")
    ap.add_argument("--date", help="UTC date to record (default: today)")
    args = ap.parse_args(argv)

    from .fetch import LiveClient

    try:
        with LiveClient() as client:
            summary = run(DATA, client, date=args.date, force=args.force)
    except Exception as e:   # login expiry, no leaderboard: fatal, nothing written
        print(f"leaderboard update FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    for w in summary["warnings"]:
        print(f"WARNING: {w}", file=sys.stderr)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

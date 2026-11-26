"""Pure rating-tracking functions. No I/O, no network.

Tracking window (user rule): a player's history gets an entry only on days they are on
the board (in the snapshot, i.e. top TOP_N). Off-board days accrue nothing. A player who
drops off and returns keeps the same history list; their streak restarts and times_entered
goes up by one. Peak, records and first_seen persist.

Every stat is derived from `history` (not incremented), so applying the same snapshot twice
gives identical output. Dates are ISO strings (YYYY-MM-DD, UTC).
"""

from __future__ import annotations

from datetime import date as _date, timedelta

TOP_TEN = 10
MOVERS_N = 5


# --- date helpers ------------------------------------------------------------------------

def _d(s: str) -> _date:
    return _date.fromisoformat(s)


def _shift(s: str, days: int) -> str:
    return (_d(s) + timedelta(days=days)).isoformat()


def _days_between(a: str, b: str) -> int:
    return (_d(b) - _d(a)).days


# --- players -----------------------------------------------------------------------------

def new_player(uid: str, username: str) -> dict:
    """A fresh record with every key present (stable schema)."""
    return {
        "uid": uid, "username": username,
        "first_seen": None, "last_seen": None,
        "currently_on_board": False, "rank": None, "rating": None,
        "left_date": None, "left_rating": None,
        "history": [],
        "peak_rating": None, "peak_rating_date": None,
        "peak_rank": None, "peak_rank_date": None,
        "days_on_board": 0, "times_entered": 0,
        "longest_streak_days": 0, "current_streak_days": 0,
        "days_in_top10": 0, "days_at_number_one": 0,
        "biggest_gain_1d": None, "biggest_drop_1d": None,
        "biggest_gain_7d": None, "biggest_drop_7d": None,
        "change_1d": None, "change_7d": None, "change_30d": None,
        "rank_change_1d": None,
    }


def _runs(hist: list) -> list[list]:
    """Split history into runs of consecutive calendar days (a run = one stay on the board)."""
    runs: list[list] = []
    cur: list = []
    for h in hist:
        if cur and _days_between(cur[-1][0], h[0]) == 1:
            cur.append(h)
        else:
            if cur:
                runs.append(cur)
            cur = [h]
    if cur:
        runs.append(cur)
    return runs


def _baseline(hist: list, before: str, days: int):
    """Latest on-board entry dated on or before `before - days` (None if there is none)."""
    cutoff = _shift(before, -days)
    best = None
    for h in hist:
        if h[0] <= cutoff:
            best = h
    return best


def _reigns(hist: list) -> list[dict]:
    """Each stretch of consecutive #1 days: {start, end, days}."""
    out = []
    for run in _runs(hist):
        cur = []
        for h in run + [None]:
            if h is not None and h[1] == 1:
                cur.append(h)
                continue
            if cur:
                out.append({"start": cur[0][0], "end": cur[-1][0], "days": len(cur)})
                cur = []
    return out


def recompute(rec: dict, date: str) -> None:
    """Set every derived field of `rec` from its history, as of `date`."""
    hist = rec["history"]
    if not hist:
        return
    runs = _runs(hist)
    last_date, last_rank, last_rating = hist[-1]
    on_board = last_date == date

    rec["first_seen"] = hist[0][0]
    rec["last_seen"] = last_date
    rec["days_on_board"] = len(hist)
    rec["times_entered"] = len(runs) - 1
    rec["longest_streak_days"] = max(len(r) for r in runs)
    rec["current_streak_days"] = len(runs[-1]) if on_board else 0
    rec["days_in_top10"] = sum(1 for h in hist if h[1] <= TOP_TEN)
    rec["days_at_number_one"] = sum(1 for h in hist if h[1] == 1)

    # peaks: highest rating and best rank; the earliest date wins a tie
    peak_r = peak_k = None
    for h in hist:
        if peak_r is None or h[2] > peak_r[2]:
            peak_r = h
        if peak_k is None or h[1] < peak_k[1]:
            peak_k = h
    rec["peak_rating"], rec["peak_rating_date"] = peak_r[2], peak_r[0]
    rec["peak_rank"], rec["peak_rank_date"] = peak_k[1], peak_k[0]

    # largest 1-day moves (consecutive on-board days only)
    gains, drops = [], []
    for run in runs:
        for prev, cur in zip(run, run[1:]):
            diff = cur[2] - prev[2]
            (gains if diff > 0 else drops if diff < 0 else []).append((diff, cur[0]))
    rec["biggest_gain_1d"] = _extreme(gains, max)
    rec["biggest_drop_1d"] = _extreme(drops, min)

    # largest 7-day moves, comparing each day with the latest on-board day 7+ days earlier
    g7, d7 = [], []
    for h in hist:
        base = _baseline(hist, h[0], 7)
        if base is not None:
            diff = h[2] - base[2]
            (g7 if diff > 0 else d7 if diff < 0 else []).append((diff, h[0]))
    rec["biggest_gain_7d"] = _extreme(g7, max)
    rec["biggest_drop_7d"] = _extreme(d7, min)

    # current-day change fields: only when on the board today
    if on_board:
        rec["rank"], rec["rating"] = last_rank, last_rating
        rec["currently_on_board"] = True
        rec["left_date"] = rec["left_rating"] = None
        prev_day = hist[-2] if len(hist) >= 2 and hist[-2][0] == _shift(date, -1) else None
        rec["rank_change_1d"] = (prev_day[1] - last_rank) if prev_day else None
        rec["change_1d"] = (last_rating - prev_day[2]) if prev_day else None
        for n, key in ((7, "change_7d"), (30, "change_30d")):
            base = _baseline(hist, date, n)
            rec[key] = (last_rating - base[2]) if base else None
    else:
        rec["rank"] = rec["rating"] = rec["rank_change_1d"] = None
        rec["change_1d"] = rec["change_7d"] = rec["change_30d"] = None
        rec["currently_on_board"] = False
        rec["left_date"] = _shift(last_date, 1)
        rec["left_rating"] = last_rating


def _extreme(items: list, pick):
    """The (diff, date) with the largest (max) or smallest (min) diff, earliest date on ties."""
    if not items:
        return None
    best = pick(items, key=lambda x: x[0])
    same = sorted(x for x in items if x[0] == best[0])
    return {"rating": same[0][0], "date": same[0][1]}


def apply_snapshot(players: dict, snapshot: dict, date: str, prev_date: str | None = None):
    """Add one day's board to `players` (keyed by uid). Returns (players, info).

    snapshot: {"players": [{"rank", "uid", "username", "rating"}, ...]}
    prev_date: the previous snapshot's date (None on the first run). Used for entered/left.
    info: {"entered": [{uid, username}], "left": [...], "movers": {"up": [...], "down": [...]}}
    Re-applying the same snapshot and date replaces that day's entry, so the result is the same.
    """
    on_board: dict[str, dict] = {}
    for row in snapshot["players"]:
        uid = str(row["uid"])
        on_board[uid] = row
        rec = players.get(uid) or new_player(uid, str(row["username"]))
        rec["username"] = str(row["username"])
        hist = [h for h in rec["history"] if h[0] != date]
        hist.append([date, int(row["rank"]), int(row["rating"])])
        hist.sort(key=lambda h: h[0])
        rec["history"] = hist
        players[uid] = rec

    entered, left = [], []
    for uid, rec in players.items():
        if uid in on_board:
            if prev_date is not None and not any(h[0] == prev_date for h in rec["history"]):
                entered.append(uid)
        elif prev_date is not None and any(h[0] == prev_date for h in rec["history"]):
            left.append(uid)
        recompute(rec, date)

    movers = _movers(players, date)
    info = {
        "entered": _listing(players, entered),
        "left": _listing(players, left),
        "movers": movers,
    }
    return players, info


def _listing(players: dict, uids: list) -> list[dict]:
    rows = [{"uid": u, "username": players[u]["username"]} for u in uids]
    return sorted(rows, key=lambda r: r["username"].lower())


def _movers(players: dict, date: str) -> dict:
    moved = [(rec, rec["change_1d"]) for rec in players.values()
             if rec["currently_on_board"] and rec["change_1d"] is not None]

    def row(rec, change):
        return {"uid": rec["uid"], "username": rec["username"], "rank": rec["rank"],
                "rating": rec["rating"], "change": change}

    ups = sorted((m for m in moved if m[1] > 0), key=lambda m: (-m[1], m[0]["username"].lower()))
    downs = sorted((m for m in moved if m[1] < 0), key=lambda m: (m[1], m[0]["username"].lower()))
    return {"up": [row(r, c) for r, c in ups[:MOVERS_N]],
            "down": [row(r, c) for r, c in downs[:MOVERS_N]]}


# --- global records ----------------------------------------------------------------------

def _pick_max(items: list, key):
    """Item with the largest key; ties go to the earlier item in `items`."""
    best = None
    for it in items:
        if best is None or key(it) > key(best):
            best = it
    return best


def build_records(players: dict, date: str, top_n: int, movers: dict | None = None) -> dict:
    """Global records as of `date`, all derived from the player records."""
    recs = list(players.values())
    out: dict = {"date": date, "top_n": top_n}

    with_peak = [r for r in recs if r["peak_rating"] is not None]
    if with_peak:
        best = _pick_max(with_peak, key=lambda r: (r["peak_rating"], -_days_between("1970-01-01", r["peak_rating_date"])))
        out["all_time_high"] = {"uid": best["uid"], "username": best["username"],
                                "rating": best["peak_rating"], "date": best["peak_rating_date"]}
    else:
        out["all_time_high"] = None

    # reigns: longest ever, and the current #1 (with the length of the reign so far)
    reigns = []
    for r in recs:
        for reign in _reigns(r["history"]):
            reigns.append({"uid": r["uid"], "username": r["username"], **reign})
    out["longest_reign"] = _pick_max(reigns, key=lambda x: (x["days"], -_days_between("1970-01-01", x["start"]))) if reigns else None
    # a reign that ends on `date` means that player is #1 on `date`
    out["number_one"] = next((dict(x) for x in reigns if x["end"] == date), None)

    out["most_days_at_number_one"] = _count_record(recs, "days_at_number_one")
    out["most_days_in_top10"] = _count_record(recs, "days_in_top10")

    for key in ("biggest_gain_1d", "biggest_drop_1d", "biggest_gain_7d", "biggest_drop_7d"):
        out[key] = _move_record(recs, key)

    out["avg_series"] = _avg_series(recs, top_n)
    out["movers"] = movers or {"up": [], "down": []}
    return out


def _count_record(recs: list, field: str):
    best = _pick_max([r for r in recs if r[field]], key=lambda r: r[field])
    if best is None:
        return None
    return {"uid": best["uid"], "username": best["username"], "days": best[field]}


def _move_record(recs: list, field: str):
    cands = [(r, r[field]) for r in recs if r[field]]
    if not cands:
        return None
    if field.startswith("biggest_gain"):
        best = _pick_max(cands, key=lambda c: c[1]["rating"])
    else:
        best = min(cands, key=lambda c: c[1]["rating"])
    rec, mv = best
    return {"uid": rec["uid"], "username": rec["username"], "rating": mv["rating"], "date": mv["date"]}


def _avg_series(recs: list, top_n: int) -> list:
    """[date, avg rating of #1, avg of top 10, avg of the whole board] for every day on record."""
    by_day: dict[str, dict[str, list[int]]] = {}
    for r in recs:
        for d, rank, rating in r["history"]:
            bucket = by_day.setdefault(d, {"one": [], "ten": [], "all": []})
            bucket["all"].append(rating)
            if rank <= TOP_TEN:
                bucket["ten"].append(rating)
            if rank == 1:
                bucket["one"].append(rating)

    def avg(xs):
        return round(sum(xs) / len(xs), 1) if xs else None

    return [[d, avg(b["one"]), avg(b["ten"]), avg(b["all"])] for d, b in sorted(by_day.items())]

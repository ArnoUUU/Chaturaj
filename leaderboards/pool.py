"""Pure pool logic: which players to probe, what the probes say, and how the board is built.

pool.json maps uid -> {uid, username, last_rank, last_rating, last_probed, in_pool}.
in_pool is true when the last probe put the player at rank <= POOL_MAX_RANK. Pool members are
probed every run. Other candidates (watch list, saved games) are probed only if they were not
probed in the last REPROBE_DAYS days, so a strong player is noticed without re-probing everyone.
"""

from __future__ import annotations

from datetime import date as _date

from .config import POOL_MAX_RANK, REPROBE_DAYS


def _days(a: str, b: str) -> int:
    return (_date.fromisoformat(b) - _date.fromisoformat(a)).days


def _recently_probed(pool: dict, uid: str, today: str) -> bool:
    last = (pool.get(uid) or {}).get("last_probed")
    return bool(last) and 0 <= _days(last, today) < REPROBE_DAYS


def gather_candidates(pool: dict, leaders: list[dict], on_board: list[dict], watch: list[dict],
                      scraper: list[dict], today: str, seed_min: int,
                      max_new: int) -> list[dict]:
    """Ordered [{uid, username, source}] to probe, without duplicates.

    Order (so a cap cuts the least important first): pool members, players on the board
    yesterday, the server's top 20, watch-list players, saved-game players. Watch-list and
    saved-game players are added only if they rate at least `seed_min`. Saved-game players are
    also skipped if probed within REPROBE_DAYS, and at most `max_new` of them are added.
    """
    out: list[dict] = []
    seen: set[str] = set()

    def add(uid, username, source) -> bool:
        uid = str(uid)
        if not uid or uid in seen or not username:
            return False
        seen.add(uid)
        out.append({"uid": uid, "username": str(username), "source": source})
        return True

    for uid in sorted(pool, key=lambda u: (pool[u].get("last_rank") or 10**9, u)):
        if pool[uid].get("in_pool"):
            add(uid, pool[uid].get("username"), "pool")
    for row in on_board:
        add(row["uid"], row["username"], "on_board")
    for row in leaders:
        add(row["uid"], row["username"], "leader")

    # watch-list players are live right now, so they are always probed (a returning drop-out
    # must be noticed on the day they reappear, not a week later)
    for row in sorted(watch, key=lambda r: -float(r.get("rating") or 0)):
        if float(row.get("rating") or 0) >= seed_min:
            add(row["uid"], row["username"], "watch")

    new = 0
    for row in sorted(scraper, key=lambda r: -float(r.get("rating") or 0)):
        if new >= max_new:
            break
        if float(row.get("rating") or 0) < seed_min or _recently_probed(pool, str(row["uid"]), today):
            continue
        if str(row["uid"]) not in seen and add(row["uid"], row["username"], "saved_game"):
            new += 1
    return out


def apply_probes(pool: dict, results: list[dict], today: str) -> dict:
    """Record each probe result. Players that were not probed keep their old entry."""
    for r in results:
        uid = str(r["uid"])
        rank = r.get("rank")
        pool[uid] = {
            "uid": uid,
            "username": r["username"],
            "last_rank": rank,
            "last_rating": r.get("rating"),
            "last_probed": today,
            "in_pool": rank is not None and rank <= POOL_MAX_RANK,
        }
    return pool


def pool_members(pool: dict) -> list[dict]:
    return sorted((v for v in pool.values() if v.get("in_pool")), key=lambda v: v.get("last_rank") or 0)


def build_board(results: list[dict], leaders: list[dict], top_n: int) -> list[dict]:
    """Board rows [{rank, uid, username, rating}] for ranks 1..top_n.

    The server's top-20 list is authoritative for its own players. Probed players are used when
    their server rank is within top_n. Ranks are the server's ranks, not a sort by rating.
    """
    rows: dict[str, dict] = {}
    for r in results:
        if r.get("rank") is None or not (1 <= r["rank"] <= top_n) or r.get("rating") is None:
            continue
        rows[str(r["uid"])] = {"rank": int(r["rank"]), "uid": str(r["uid"]),
                               "username": r["username"], "rating": int(r["rating"])}
    for r in leaders:
        rows[str(r["uid"])] = {"rank": int(r["rank"]), "uid": str(r["uid"]),
                               "username": r["username"], "rating": int(r["rating"])}
    return sorted(rows.values(), key=lambda r: (r["rank"], r["username"].lower()))

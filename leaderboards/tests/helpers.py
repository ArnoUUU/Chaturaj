"""Shared test helpers: a mock client that plays the server's role.

`truth` maps a date to the full ordering of the board that day: [(uid, username, rating), ...]
in rank order. The mock serves:
  leaders()  the first `leader_n` players (the server only shows the top 20 in reality)
  watch()    every player in the day's truth (so the pool can discover them)
  probe()    each candidate's true rank (None when not on the board), or raises ProbeError
"""

from __future__ import annotations

import json
from pathlib import Path

from leaderboards.prober import ProbeError

UIDS = {"A": "101", "B": "102", "C": "103", "D": "104", "X": "105"}
DATES = [f"2026-11-0{i}" for i in range(1, 6)]


class MockClient:
    def __init__(self, truth: dict, leader_n: int = 1, fail_probe_days=(), scraper_rows=()):
        self.truth = truth
        self.day = None
        self.leader_n = leader_n
        self.fail_probe_days = set(fail_probe_days)
        self.scraper_rows = list(scraper_rows)
        self.probe_calls: list[list[str]] = []

    def _cur(self):
        return self.truth[self.day]

    def leaders(self):
        return [{"rank": i, "uid": u, "username": n, "rating": r}
                for i, (u, n, r) in enumerate(self._cur()[:self.leader_n], 1)]

    def watch(self):
        return [{"uid": u, "username": n, "rating": float(r)} for (u, n, r) in self._cur()]

    def scraper(self):
        return list(self.scraper_rows)

    def probe(self, cands, delay: float = 0.0):
        if self.day in self.fail_probe_days:
            raise ProbeError("mock: connection lost")
        self.probe_calls.append([c["uid"] for c in cands])
        cur = {u: (i + 1, r) for i, (u, _, r) in enumerate(self._cur())}
        names = {u: n for (u, n, _) in self._cur()}
        results = []
        for c in cands:
            rank, rating = cur.get(c["uid"], (None, None))
            results.append({"uid": c["uid"], "username": names.get(c["uid"], c["username"]),
                            "rank": rank, "rating": rating})
        return results, []


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def five_day_truth(top: int = 4) -> dict:
    """The stats scenario, as a full ordering per day (top 4 shown by the tests)."""
    sched = {
        DATES[0]: [("A", 2000), ("B", 1900), ("X", 1800), ("C", 1700)],
        DATES[1]: [("X", 1850), ("A", 2010), ("B", 1900), ("C", 1690)],
        DATES[2]: [("A", 2020), ("B", 1905), ("C", 1700), ("D", 1650)],
        DATES[3]: [("A", 2020), ("B", 1880), ("C", 1700), ("D", 1660)],
        DATES[4]: [("A", 2030), ("B", 1880), ("X", 1700), ("D", 1660)],
    }
    return {d: [(UIDS[n], n, r) for n, r in rows] for d, rows in sched.items()}

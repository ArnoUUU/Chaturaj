"""Per-player rank probe over the variants socket (read-only).

The profile card on chess.com shows a player's rank and rating per variant. The client
sends  player-popup {playerId, username, contextQ}  and the server replies with mutation
`user_popup`, whose `compositeRatings[<board>]` holds {rating, lbRank}. For the Chaturaji
blitz board that key is `rt:chaturaji:blitz`. lbRank is the server's rank on that board,
so the probe gives an exact rank for any player it is asked about.

Only this one request type is sent. No seeks, joins, or games.
"""

from __future__ import annotations

import time

from .config import BOARD

POPUP_ACTION = "player-popup"
POPUP_MUTATION = "user_popup"
PROBE_TIMEOUT = 8.0


class ProbeError(RuntimeError):
    """The probe cannot continue (connection lost, too many failures). The daily job falls back."""


def board_entry(popup: dict, board: str = BOARD) -> dict:
    """{rank, rating} for `board` from a user_popup reply. rank is None when the server has none."""
    comp = (popup.get("compositeRatings") or {}).get(board) or {}
    rating = comp.get("rating")
    rank = comp.get("lbRank")
    return {
        "rank": int(rank) if isinstance(rank, (int, float)) and rank > 0 else None,
        "rating": round(float(rating)) if isinstance(rating, (int, float)) else None,
    }


def probe_one(sock, uid: str, username: str, board: str = BOARD) -> dict:
    """Ask the server for one player's rank and rating on `board`."""
    sock.send(POPUP_ACTION, {"playerId": int(uid), "username": username, "contextQ": None})

    def is_reply(mutation, data):
        return mutation == POPUP_MUTATION and isinstance(data, dict) and str(data.get("uid")) == str(uid)

    _, popup = sock.wait(is_reply, timeout=PROBE_TIMEOUT)
    return {"uid": str(uid), "username": popup.get("username", username), **board_entry(popup, board)}


def probe_many(sock, players: list[dict], delay: float = 1.0, board: str = BOARD,
               max_fail_ratio: float = 0.3) -> tuple[list[dict], list[dict]]:
    """Probe each {uid, username}; returns (results, failures).

    Sleeps `delay` seconds between players. A single timeout is recorded as a failure and the
    probe continues. Raises ProbeError when the connection drops or more than `max_fail_ratio`
    of the players time out (so the daily job can fall back instead of writing bad ranks).
    """
    results, failures = [], []
    for i, p in enumerate(players):
        try:
            results.append(probe_one(sock, p["uid"], p["username"], board))
        except TimeoutError:
            failures.append({"uid": str(p["uid"]), "username": p["username"], "error": "timeout"})
        except Exception as e:   # connection-level errors end the probe
            raise ProbeError(f"socket error after {i} players: {type(e).__name__}: {e}") from e
        if len(failures) > max_fail_ratio * (i + 1) and len(failures) >= 5:
            raise ProbeError(f"{len(failures)} of {i + 1} probes timed out")
        if i + 1 < len(players):
            time.sleep(delay)
    return results, failures

"""Phase 0: discover the chess.com Chaturaji leaderboard action.

Read-only. Makes a handful of HTTP requests (the variant pages and their public JS
bundles) and sends a short list of read-only candidate actions over the variants
WebSocket. Never sends seek, join, or game-start messages. Everything it sees is
written to leaderboards/probe_out/ (gitignored).

Run from the repo root:
    python leaderboards/probe.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = Path(__file__).resolve().parent / "probe_out"
PAGES = [
    "https://www.chess.com/variants/chaturaji",
    "https://www.chess.com/variants/chaturaji/leaderboard",
    "https://www.chess.com/variants/chaturaji/players",
]
KEYWORDS = re.compile(r"leaderboard|ranking|rating[s]?\b|top[-_ ]?(players|100)|standings", re.I)
CANDIDATE_ACTIONS = [
    "leaderboard", "get-leaderboard", "leaderboards", "get-leaderboards",
    "ratings", "get-ratings", "top-players", "get-top-players",
    "rankings", "get-rankings", "players", "get-players",
]
CANDIDATE_PAYLOADS = [
    {},
    {"variant": "chaturaji"},
    {"variantUrl": "chaturaji", "ratingType": "", "limit": 100, "page": 0},
]
LISTEN_SECONDS = 3.0
SLEEP_BETWEEN = 1.0   # polite pause between socket requests


def save(name: str, data) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    if isinstance(data, (dict, list)):
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    else:
        path.write_text(str(data), encoding="utf-8")
    print(f"  saved {path}")


def probe_http() -> list[str]:
    """Fetch the variant pages with the saved login; return the JS bundle URLs found."""
    from scraper import ws
    http = ws.VariantsSocket()._http()
    scripts: set[str] = set()
    for url in PAGES:
        r = http.get(url, timeout=15)
        print(f"GET {url} -> {r.status_code} ({len(r.text)} bytes)")
        slug = re.sub(r"[^a-z0-9]+", "_", url.split("chess.com")[1].lower()).strip("_") or "root"
        save(f"page_{slug}.html", r.text)
        for src in re.findall(r'<script[^>]+src="([^"]+)"', r.text):
            scripts.add(src if src.startswith("http") else "https://www.chess.com" + src)
        time.sleep(1.0)
    return sorted(scripts)


def probe_bundles(urls: list[str]) -> None:
    """Search the public JS bundles for leaderboard-related strings (snippets only)."""
    import requests
    hits: dict[str, list[str]] = {}
    for url in urls[:40]:
        if "chess.com" not in url:
            continue
        try:
            js = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"}).text
        except requests.RequestException as e:
            print(f"  skip {url}: {e}")
            continue
        time.sleep(0.5)
        snippets = []
        for m in KEYWORDS.finditer(js):
            snippets.append(js[max(0, m.start() - 120): m.end() + 120].replace("\n", " "))
            if len(snippets) >= 25:
                break
        if snippets:
            hits[url] = snippets
    save("bundle_hits.json", hits)
    actions = sorted({a for s in hits.values() for snip in s
                      for a in re.findall(r'"((?:get-)?(?:leaderboard|ranking|rating|top)[\w-]*)"', snip)})
    print(f"action-like strings in bundles: {actions}")


def probe_socket() -> None:
    from scraper import ws
    results = []
    with ws.VariantsSocket() as sock:
        print(f"connected as {sock.user.get('username')}")
        for action in CANDIDATE_ACTIONS:
            for payload in CANDIDATE_PAYLOADS:
                sock.send(action, payload)
                got = []
                end = time.time() + LISTEN_SECONDS
                while time.time() < end:
                    try:
                        mutation, data = sock.recv()
                    except Exception as e:   # timeout on a quiet socket is normal here
                        got.append({"error": type(e).__name__})
                        break
                    if mutation in ("ping",):
                        continue
                    got.append({"mutation": mutation, "data": data})
                # only keep replies that look like a list of players/ratings
                results.append({"action": action, "payload": payload,
                                "replies": [g.get("mutation") for g in got]})
                for g in got:
                    if g.get("mutation") and g["mutation"] not in ("ping", "user_connected"):
                        print(f"  {action} {payload} -> {g['mutation']}")
                save(f"socket_{action}_{len(results)}.json", got)
                time.sleep(SLEEP_BETWEEN)
    save("socket_summary.json", results)


ROW_PAYLOADS = [
    {"leaderboardType": "rt:chaturaji:blitz"},
    {"leaderboardType": "rt:chaturaji:blitz", "page": 1},
    {"leaderboardType": "rt:chaturaji:blitz", "limit": 100},
    {"leaderboardType": "rt:chaturaji:blitz", "offset": 20, "limit": 20},
]


def probe_rows() -> None:
    """Find the message that returns the Chaturaji board rows (and how paging works)."""
    from scraper import ws
    with ws.VariantsSocket() as sock:
        print(f"connected as {sock.user.get('username')}")
        for i, payload in enumerate(ROW_PAYLOADS):
            sock.send("leaderboard", payload)
            got = []
            end = time.time() + 6.0
            while time.time() < end:
                try:
                    mutation, data = sock.recv()
                except Exception as e:
                    got.append({"error": type(e).__name__})
                    break
                if mutation != "ping":
                    got.append({"mutation": mutation, "data": data})
            print(f"  {payload} -> {[g.get('mutation') or g.get('error') for g in got]}")
            save(f"rows_{i}.json", got)
            time.sleep(2.0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if "rows" in sys.argv[1:]:
        probe_rows()
        return
    try:
        scripts = probe_http()
    except Exception as e:
        print(f"HTTP probe failed: {type(e).__name__}: {e}")
        return
    save("script_urls.json", scripts)
    probe_bundles(scripts)
    try:
        probe_socket()
    except Exception as e:
        print(f"socket probe failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

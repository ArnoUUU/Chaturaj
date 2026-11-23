"""Piece sprites for the viewer, taken from chess.com's 4-player/variants piece set (personal use).

chess.com draws variant pieces from SVG templates inside its public variants script
(theme "neo"), with a ___COLORCODE___ placeholder for the seat colour. This module
downloads that script once, extracts the five templates Chaturaji uses (pawn, knight,
bishop, king, and the sailboat used for the rook) and caches them in
gui/.cache/sprites.json, which is not committed. If the download fails, simple built-in
fallback sprites are used so the viewer still works.

    python -m gui.sprites            # download / refresh the cache
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from xml.etree import ElementTree

CACHE = Path(__file__).resolve().parent / ".cache" / "sprites.json"
VARIANTS_PAGE = "https://www.chess.com/variants/chaturaji"
SCRIPT_RE = re.compile(r"(?:https://www\.chess\.com)?/r2/client-packages/variants/[^\"'\s]+/variants\.js")
# engine piece letter -> chess.com template name (Chaturaji rooks are drawn as boats)
PIECE_TEMPLATES = {"P": "Pawn", "N": "Knight", "B": "Bishop", "K": "King", "R": "Sailboat"}
# chess.com seat colours, and the grey used for dead pieces
COLORS = {"R": "#BF3B43", "B": "#4185BF", "Y": "#C09526", "G": "#4E9161", "X": "#BDBCB8"}


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def extract_templates(script: str) -> dict:
    """Pull the SVG wrapper and the neo-theme templates out of chess.com's variants.js."""
    wrap = re.search(r"wrapper:\{head:`(.*?)`,foot:\"(.*?)\"\}", script, re.S)
    if not wrap:
        raise ValueError("SVG wrapper not found in variants.js")
    themes = script.index("themes:{")
    neo = re.compile(r"\[\w{1,3}\.neo\]:\{").search(script, themes)
    if not neo:
        raise ValueError("neo piece theme not found in variants.js")
    templates = {}
    for letter, name in PIECE_TEMPLATES.items():
        m = re.compile(r"\[\w{1,3}\." + name + r"\]:'").search(script, neo.end())
        if not m:
            raise ValueError(f"{name} template not found")
        templates[letter] = script[m.end():script.index("'", m.end())]
    return {"head": wrap.group(1), "foot": wrap.group(2), "pieces": templates}


def download(refresh: bool = False) -> dict:
    if CACHE.exists() and not refresh:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    page = _get(VARIANTS_PAGE)
    m = SCRIPT_RE.search(page)
    if not m:
        raise ValueError("variants.js link not found on the chess.com variants page")
    url = m.group(0) if m.group(0).startswith("http") else "https://www.chess.com" + m.group(0)
    data = extract_templates(_get(url))
    data["source"] = url
    for letter in PIECE_TEMPLATES:  # make sure every sprite is valid SVG before caching
        ElementTree.fromstring(render(data, letter, "#000000"))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(data), encoding="utf-8")
    return data


def render(data: dict, letter: str, color: str) -> str:
    return data["head"] + data["pieces"][letter].replace("___COLORCODE___", color) + data["foot"]


def fallback_svg(letter: str, color: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 180 180">'
            f'<circle cx="90" cy="90" r="70" fill="{color}" stroke="#222" stroke-width="6"/>'
            f'<text x="90" y="118" font-size="84" font-family="sans-serif" font-weight="bold" '
            f'text-anchor="middle" fill="#fff">{letter}</text></svg>')


def sprite_set(refresh: bool = False) -> tuple[dict[str, str], str]:
    """{'RK': svg, 'XP': svg, ...} for every owner/piece, plus where they came from."""
    try:
        data = download(refresh)
        make, source = (lambda l, c: render(data, l, c)), "chess.com"
    except Exception as e:  # offline, or chess.com changed its script
        make, source = fallback_svg, f"fallback ({e})"
    return {o + l: make(l, c) for o, c in COLORS.items() for l in PIECE_TEMPLATES}, source


if __name__ == "__main__":
    sprites, source = sprite_set(refresh=True)
    print(f"{len(sprites)} sprites from {source}; cached at {CACHE}")

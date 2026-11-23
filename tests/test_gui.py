import json

from gui import sprites, viewer
from scraper.pgn4 import to_game
from tests.test_pgn4 import REAL_GAME_113036725


def test_extract_templates_from_script_shape():
    script = ("x={wrapper:{head:`<svg xmlns=\"http://www.w3.org/2000/svg\">`,foot:\"</svg>\"},"
              "themes:{[on.alpha]:{[Ye.Pawn]:'<g/>'},[on.neo]:{[Ye.Pawn]:'<rect fill=\"___COLORCODE___\"/>',"
              "[Ye.Knight]:'<g/>',[Ye.Bishop]:'<g/>',[Ye.King]:'<g/>'},"
              "[on.other]:{}},y={[Ye.Sailboat]:'<g/>'}")
    data = sprites.extract_templates(script)
    assert data["pieces"]["P"] == '<rect fill="___COLORCODE___"/>'
    assert sprites.render(data, "P", "#BF3B43").endswith('<rect fill="#BF3B43"/></svg>')


def test_viewer_builds_page_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer, "sprite_set", lambda: (
        {o + l: sprites.fallback_svg(l, c) for o, c in sprites.COLORS.items() for l in "PNBKR"}, "test"))
    g = to_game(REAL_GAME_113036725, (3, 3), {"chesscom_id": "113036725",
                                              "seats": {"Blue": {"name": "griffinlc"}}})
    g.save(tmp_path / "113036725.json")
    out = viewer.build(tmp_path, out=tmp_path / "v.html")
    html = out.read_text(encoding="utf-8")
    games = json.loads(html.split("const GAMES=")[1].split(", SPRITES=")[0])
    last = games[0]["frames"][-1]
    assert last["s"] == [13, 25, 0, 10] and len(last["b"]) == 128
    assert games[0]["names"][1] == "griffinlc"
    assert games[0]["frames"][-1]["m"] == "Red resigned"

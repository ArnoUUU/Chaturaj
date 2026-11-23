# scraper — chess.com Chaturaji games

`game_scraping.py` scrapes Chaturaji games from chess.com **by player and by time
control**. It saves each game (PGN4, ratings, move times, chat) and converts it into a
validated `chaturaji` record for the viewer and for training. `pgn4.py` holds the
PGN4 → engine conversion.

By default it talks straight to the chess.com variants WebSocket (`ws.py`), with no
browser. A browser is only needed once, to log in. The older Selenium scraper is still
there behind `--selenium`.

| | WebSocket (default) | `--selenium` |
|---|---|---|
| archive page (20 games) | ~0.1 s | several seconds of clicking and rendering |
| one game | ~0.1–0.4 s + `--delay` (1 s) | page load + `--delay` (4 s) |
| reach | whole history in one paged search | date windows + "Show more" |
| move times | ms timestamps + clock left | wall-clock seconds |
| chat | exact timestamp and seat per line | estimated from game events |

## Requirements
- `pip install requests websocket-client` for scraping.
- `pip install selenium` plus Firefox (or Chrome) for `login`, and for `--selenium`.
- A chess.com account. Chess.com only shows variant games to logged-in users, and its
  Terms of Service restrict automated access, so use your own account and keep a delay
  between games.

## Commands
Run these from the repository root.

Log in once. This opens Firefox; the login cookies are saved to
`scraper/.firefox-profile/chesscom-cookies.json`, which the WebSocket scraper reuses.
Run it again when the scraper says the login has expired:
```powershell
python -m scraper.game_scraping login
```

Scrape a player's games, saved per time control (up to 20 new games per control by default):
```powershell
python -m scraper.game_scraping player arnav2007 --time-control "1 | 5"
```
```powershell
python -m scraper.game_scraping player arnav2007 --time-control "1 | 5" --time-control "30 sec" --max-per-tc 50
```
```powershell
python -m scraper.game_scraping player arnav2007
```
(no `--time-control` means every time control). `--days "0-30"` limits the search to the
last 30 days. Add `--selenium` to any of these to use the browser instead.

Scrape specific games, optionally adding them to a player's index:
```powershell
python -m scraper.game_scraping games 113032695 113031547 --player arnav2007
```

Rebuild every engine record from the saved games (after a rules or parser change):
```powershell
python -m scraper.game_scraping convert
```

From Python:
```python
from scraper import ws
with ws.VariantsSocket() as sock:
    ws.scrape_player(sock, "arnav2007", time_controls=["1 | 5", "30 sec"], max_per_tc=20)
    raw = sock.game(113032695)          # the server's full game message
```

Time controls are written as the archive shows them: `"1 | 5"` (blitz), `"6s | 3"`,
`"30 sec"`, and so on.

## How scraped data is saved
```
scraper/data/
  games/<game_id>.json                       full scraped game, stored once even if several players are scraped
  records/<game_id>.json                     chaturaji Game record (viewer + training), rebuilt by `convert`
  players/<name>/<time_control>/index.jsonl  one line per game for that player and control
  players/<name>/<time_control>/summary.json games, wins, average place/points, ratings
  failed.jsonl                               anything that could not be scraped or converted
```
Folder names for time controls: `1 | 5` → `1+5`, `6s | 3` → `6s+3`, `30 sec` → `30s`.
Player folders are lowercase.

- **`games/<id>.json`** holds:
  - the PGN4 and its headers
  - `seats` (color → name, rating)
  - `final_points`, `termination`, `time_control`
  - `rule_variants` and `anonymity` (e.g. `SemiAnonymous`)
  - `moves`: each move with its timestamp and the seconds since the previous move
    (WebSocket: ISO UTC `time` with milliseconds, plus `clock_ms`, the mover's clock)
  - `chat`: system/player lines, each with `after_ply` / `before_ply` and their times
    (WebSocket: also the line's own `time`, `seat` and `full_move`)
  - WebSocket: `players` (seat, name, id, rating, rating change, points, place),
    `start_date`/`end_date`, `rated`, `source: "websocket"`. Selenium: `player_boxes`
    (final clocks) and `observers`.
  - `archive_row`: the list entry, including rating changes
- **`index.jsonl`** has one line per game, with `game_id`, `date`, `seat`, `rating`,
  `rating_change`, `points`, `place` (1–4), `opponents`, `termination`, `anonymity`,
  `plies`, `chat_messages`, and `record` (whether the engine replay succeeded).
- **Re-runs are incremental.** Games already in a player's index are skipped before they're
  opened (WebSocket: by game id; Selenium: by date and the four player names). A game
  already saved for another player is just added to the index, not downloaded again.
  Use `--refetch` to redo them.

## How it works
- **WebSocket protocol** (see the docstring in `ws.py`): an HTTP request with the saved
  cookies fetches the logged-in user object chess.com embeds in its pages. That is sent as
  `connect-user`, then `autocomplete-username` turns a name into a player id,
  `search-archive` returns 20 games per page, and `game` / `join` returns the stored game
  with a PGN4 whose move comments hold `date=` and `clock=`. Player IP hashes in the
  server's messages are not saved.
- **Selenium archive search:** chess.com only runs the archive search when a username suggestion
  is clicked, so the script types the name and clicks the suggestion itself.
- **Selenium, more than 20 games:** each search shows about 20 games, so the script walks through
  date ranges (`0-1`, `0-7`, `0-14`, `14-31`, … `365-9999`) and clicks "Show more" until
  each requested time control has enough games. Chess.com's own time-control filter
  doesn't change the results, so filtering happens in the script.
- **Engine conversion:** each game's PGN4 is replayed through the rules engine
  (`pgn4.to_game`, with the 14x14 → 8x8 offset), which checks every move is legal.
  The engine's scoring has matched chess.com's final scores on every game checked.
- **Selenium chat timing:** chess.com shows no timestamps on chat messages, but every move has one.
  Each message is placed between the nearest timed game events around it in the chat
  (checkmated, meaning a king was captured; resigned; timed out; claimed the win).

## Tests
```powershell
python -m pytest -q tests/test_ws.py tests/test_game_scraping.py tests/test_pgn4.py
```

# scraper — chess.com Chaturaji games

`game_scraping.py` scrapes Chaturaji games from chess.com with Selenium, **by player and by
time control**. It saves everything shown on each game page and converts every game into
a validated `chaturaji` record for the viewer and for training. `pgn4.py` holds the
PGN4 → engine conversion.

## Requirements
- `pip install selenium`, plus Google Chrome. Selenium 4 downloads the matching driver.
- A chess.com account. Chess.com only shows variant games to logged-in users, and its
  Terms of Service restrict automated access, so use your own account and keep the
  default delays (about 4–6 seconds per game).

## Commands
Run these from the repository root.

Log in once (the session is kept in `scraper/.selenium-profile/`):
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
(no `--time-control` means every time control)

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
from scraper.game_scraping import make_driver, scrape_player
driver = make_driver()
scrape_player(driver, "arnav2007", time_controls=["1 | 5", "30 sec"], max_per_tc=20)
driver.quit()
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
  - `chat`: system/player lines, each with `after_ply` / `before_ply` and their times
  - `player_boxes` (final clocks), `observers`
  - `archive_row`: the list entry, including rating changes
- **`index.jsonl`** has one line per game, with `game_id`, `date`, `seat`, `rating`,
  `rating_change`, `points`, `place` (1–4), `opponents`, `termination`, `anonymity`,
  `plies`, `chat_messages`, and `record` (whether the engine replay succeeded).
- **Re-runs are incremental.** Games already in a player's index are skipped before they're
  opened, matched by date and the four player names. Use `--refetch` to redo them.

## How it works
- **Archive search:** chess.com only runs the archive search when a username suggestion
  is clicked, so the script types the name and clicks the suggestion itself.
- **More than 20 games:** each search shows about 20 games, so the script walks through
  date ranges (`0-1`, `0-7`, `0-14`, `14-31`, … `365-9999`) and clicks "Show more" until
  each requested time control has enough games. Chess.com's own time-control filter
  doesn't change the results, so filtering happens in the script.
- **Engine conversion:** each game's PGN4 is replayed through the rules engine
  (`pgn4.to_game`, with the 14x14 → 8x8 offset), which checks every move is legal.
  The engine's scoring has matched chess.com's final scores on every game checked.
- **Chat timing:** chess.com shows no timestamps on chat messages, but every move has one.
  Each message is placed between the nearest timed game events around it in the chat
  (checkmated, meaning a king was captured; resigned; timed out; claimed the win).

## Tests
```powershell
python -m pytest -q tests/test_game_scraping.py tests/test_pgn4.py
```

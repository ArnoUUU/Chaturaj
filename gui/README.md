# gui — play and view Chaturaji on your own machine

Two small tools with no packages beyond the repo's own code and NumPy/PyTorch for the bots.

| Command | What it does |
|---|---|
| `python -m gui.play` | Play timed games against the bots in your browser |
| `python -m gui.viewer <file or folder>` | Replay recorded games, one move at a time |

Everything runs on your machine. The play server listens on `127.0.0.1` only.

## Play

```bash
python -m gui.play                                 # opens the setup page
python -m gui.play --human R B                     # two people at one screen, taking turns
python -m gui.play --human none --no-open          # bots only, played out in the terminal
python -m gui.play --seat R=human B=uniform Y=models/opponent_pool/BEST_taran0915_model.pt G=human
```

The page opens on a setup screen:

- **A dropdown per side.** Each side is `Human` or a bot: the pool models (`models/opponent_pool/`),
  any model listed in `models/index.jsonl`, or `Uniform` (no model, a weak baseline). The default
  bot is the benchmark model, else the best scored model.
- **Time control.** `minutes|increment`, for example `1|5` (1 minute, +5 s per move). Presets are
  offered in the box; any value such as `0.5|2` also works.
- **Start game.** Then the board appears.

During a game:

- **Clocks** work like chess.com. Each side has its own clock, the increment is added after each
  move, and a side that runs out of time is eliminated on time (its pieces turn grey). The clock
  of the side to move counts down.
- **Turns.** When two or more people share the screen, the board can follow the side to move
  ("Board follows the side to move"), so each person sees their own pieces at the bottom on their
  turn. With one person, the board stays on their side.
- **Bots** choose how long to think from their own clock (`bot/timing.py`). The panel shows each
  bot's simulations, time, and its chances of finishing first.
- **Resign** resigns the side to move if it is a person, otherwise the first person still playing.
  Only people can resign; bots play to the end of their clock.
- **New game** returns to the setup screen (asks first if a game is in progress and stops it).

Controls during a game:

- **Pause / Resume** (or the space bar) stops the clocks and every bot's thinking. A paused
  game can still be reviewed; nobody can move until it is resumed.
- **Players.** Each side has a dropdown to change who plays it, between moves: a person, or any
  bot. A bot that was thinking for that side stops at once.
- **Review.** `|<` `<` `>` `>|` step through the moves, or click a move in the list (arrow keys
  and Home/End work too). Earlier positions show the clocks as they were after that move. Live
  moves are not possible while reviewing; press **Live** (or End) to go back.
- **Move list.** Each move shows its number, the piece-coloured side, the move (SAN), any points
  scored, and the clock after it.
- **Clock for people.** Switched off at setup, a person has no clock and cannot run out of time.
  Bots keep their clocks.
- **Time control.** A dropdown of the time controls the scraper has recorded (most played first, with
  game counts, from `scraper/data/players/`), or Custom.

Finished games are saved to `games/played/` and open in the viewer:
`python -m gui.viewer games/played`.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--human R B …` | `R` | Seats played by people (`none` for bots only) |
| `--seat COLOR=VALUE …` | | Set each side: `human`, `uniform`, or a model. Overrides `--human` and `--engine` |
| `--engine` | benchmark, else best | Bot for the other seats |
| `--time` | `1\|5` | Time control |
| `--max-sims` | none | Cap simulations per move (for a faster, weaker game) |
| `--sprites` | `chesscom` | `builtin` uses simple built-in pieces (see below) |
| `--port` | `8765` | `0` picks a free port |
| `--threads` | | Torch CPU threads for the bots |
| `--no-open` | | Don't open the browser |

## Layout

```
gui/play.py           server, game session, bot bank, HTTP API
gui/static/           the page: index.html, play.css, play.js (no build step)
gui/viewer.py         replay viewer
gui/sprites.py        piece artwork (see below)
gui/.cache/           downloaded artwork and generated pages (not committed)
tests/test_play.py    session rules, clocks, and the HTTP API
```

### HTTP API (for reference)

| Method and path | Body / result |
|---|---|
| `GET /api/bots` | `{options: [{value, label}], default}` |
| `GET /api/assets` | piece artwork, board views, clock presets |
| `POST /api/new` | `{seats: {R: "human" \| model, …}, time: "1\|5", human_clock: true}`; 400 with an error message if invalid |
| `GET /api/state[?ply=N]` | the game snapshot, or `{idle: true}`; `ply` shows an earlier position for review; the first call starts the clocks when people play |
| `POST /api/move` | `{color, from, to}` (board indexes 0–63) |
| `POST /api/resign` | `{color}` |
| `POST /api/seat` | `{color, value}`: `value` is `human`, `uniform`, or a model |
| `POST /api/pause` | `{paused: true \| false}` |
| `POST /api/close` | stops the current game |

## Piece artwork

`--sprites chesscom` (the default) uses chess.com's piece artwork, downloaded once from
chess.com's public variants script and cached in `gui/.cache/` (not committed). `--sprites builtin`
draws simple coloured discs with the piece letter, and needs no download.

**Before publishing this project, decide which artwork ships.** chess.com's artwork belongs to
chess.com. The default downloads it at run time, so the repository does not contain it, but
redistributing the artwork or relying on the download for a public release may not be allowed.
`--sprites builtin` is the safe default for a public release.

## Tests

```bash
python -m pytest tests/test_play.py -q
```

The tests use a uniform bot with a few simulations, so they need no trained model and finish in
about 15 seconds.

## Limits

- One game per server. Starting a new game closes the previous one.
- Bots run on the CPU. A long time control gives the bot more search time, so a game takes as
  long as its clocks.
- A model that is not on disk is rejected when the game starts.

# Chaturaj
A Chaturaji (4-player chess) engine: a rules engine, a PyTorch self-play bot, and a
chess.com game scraper.

| Folder | What it is |
|---|---|
| [`chaturaji/`](chaturaji/README.md) | Board state and rules: move generation, scoring, game records, notation, terminal viewer |
| [`bot/`](bot/README.md) | AlphaZero-style bot: PyTorch network, MCTS, self-play and training; the NNUE + max^n engine and its training: [bot/README_NNUE.md](bot/README_NNUE.md) |
| [`scraper/`](scraper/README.md) | Scrapes chess.com Chaturaji games by player and time control (over the chess.com variants WebSocket; Selenium only to log in) and converts them to game records |
| [`cluster/`](cluster/README.md) | GPU runs on the Tillicum cluster: Miniforge conda env, Slurm job scripts |
| [`gui/`](gui/README.md) | Browser game viewer (`python -m gui.viewer`) and timed play against bots (`python -m gui.play`) |
| [`models/`](models/README.md) | Saved, labelled bot models (one folder per training run) |
| `logs/` | Chunked run logs, GPU check reports, Slurm output (not committed) |
| `tests/` | pytest suite |

Rules and design decisions are in [PLAN.md](PLAN.md). The model plan is in [bot/PLAN.md](bot/PLAN.md).

## Setup
Python 3.10+ (developed on 3.14).

```powershell
pip install numpy torch pytest selenium
```

- `chaturaji/` needs no third-party packages. `numpy` is only used by `chaturaji/encoding.py`.
- `bot/` needs `numpy` and `torch`. The CPU build is fine for trying it out.
- `scraper/` needs `requests` and `websocket-client`, plus `selenium` and Firefox for the
  one-time login (and for the optional `--selenium` mode).

## Quick check
Run everything from the repository root:

```powershell
python -m pytest -q
```

```powershell
python -m chaturaji.simulate --games 1 --seed 0 --out games
```

```powershell
python -m chaturaji.viewer games/game_0.json
```

## Browser GUI
`gui/` runs in your browser on this machine, with no packages beyond the repo's own.

- **Play** against the bots: `python -m gui.play`. The setup screen gives each side a dropdown
  (you or a bot), a time control taken from the games the scraper recorded, and an option to
  switch off the clock for people. During a game you can change a side's player, pause and
  resume (which stops the clocks and bot thinking), and step back and forward through the moves.
- **Review** a finished game: `python -m gui.viewer games/played` (or a single game file).

Other options: `--human B` plays Blue, `--human R Y` puts two people at one screen,
`--seat R=human B=uniform` sets each side, `--engine models/<label>` picks the bot for the other
seats (default: the benchmark model, else the best evaluated one), and `--human none --no-open`
plays bots only in the terminal. Finished games are saved to `games/played/`.

Full controls, options and the HTTP API are in [gui/README.md](gui/README.md).

Generated folders (`games/`, `logs/`, `scraper/data/`, `scraper/.selenium-profile/`) are in
`.gitignore`. Saved models in `models/` are kept in git.

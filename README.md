# Chaturaj
A Chaturaji (4-player chess) engine: a rules engine, a PyTorch self-play bot, and a
chess.com game scraper.

| Folder | What it is |
|---|---|
| [`chaturaji/`](chaturaji/README.md) | Board state and rules: move generation, scoring, game records, notation, terminal viewer |
| [`bot/`](bot/README.md) | AlphaZero-style bot: PyTorch network, MCTS, self-play and training; the NNUE + max^n engine and its training: [bot/README_NNUE.md](bot/README_NNUE.md) |
| [`scraper/`](scraper/README.md) | Scrapes chess.com Chaturaji games by player and time control (over the chess.com variants WebSocket; Selenium only to log in) and converts them to game records |
| [`cluster/`](cluster/README.md) | GPU runs on the Tillicum cluster: Miniforge conda env, Slurm job scripts |
| `gui/` | Browser game viewer (`python -m gui.viewer`) and timed play against the engine (`python -m gui.play`) |
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

## Play the engine
Play a timed game (chess.com `1 | 5` by default) against the engine in your browser. You are
Red and the engine plays the other three seats:

```powershell
python -m gui.play
```

`--human B` plays Blue, `--human R Y` puts two people at one screen, `--time "3|2"` changes
the time control and `--engine models/<label>` picks the model (default: the best evaluated
one in `models/index.jsonl`). The engine decides how long to think from its own clock.
Finished games are saved to `games/played/` and replay with `python -m gui.viewer games/played`.
`python -m gui.play --human none --no-open` plays the engine against itself on the clock in
the terminal, as a check that it never loses on time.

Generated folders (`games/`, `logs/`, `scraper/data/`, `scraper/.selenium-profile/`) are in
`.gitignore`. Saved models in `models/` are kept in git.

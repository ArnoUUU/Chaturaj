# Chaturaj
A Chaturaji (4-player chess) engine: a rules engine, a PyTorch self-play bot, and a
chess.com game scraper.

| Folder | What it is |
|---|---|
| [`chaturaji/`](chaturaji/README.md) | Board state and rules: move generation, scoring, game records, notation, terminal viewer |
| [`bot/`](bot/README.md) | AlphaZero-style bot: PyTorch network, MCTS, self-play and training |
| [`scraper/`](scraper/README.md) | Scrapes chess.com Chaturaji games by player and time control (over the chess.com variants WebSocket; Selenium only to log in) and converts them to game records |
| [`cluster/`](cluster/README.md) | GPU runs on the Tillicum cluster: Miniforge conda env, Slurm job scripts |
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

Generated folders (`games/`, `logs/`, `scraper/data/`, `scraper/.selenium-profile/`) are in
`.gitignore`. Saved models in `models/` are kept in git.

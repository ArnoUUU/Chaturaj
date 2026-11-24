# engine — compiled core (Rust + PyO3)

The Python module `chaturaji_engine`. `bot/nnue.py` uses it automatically when it is
installed (`bot.nnue.CORE_AVAILABLE`), and falls back to NumPy when it isn't. Results are
the same either way; the tests check that.

## What's in it

| File | Contents |
|---|---|
| `src/nnue.rs` | NNUE inference, the same network as `bot/nnue.py`: the feature table, accumulators from scratch (`refresh`), incremental updates from a board difference (`apply_diff`), a push/pop stack for search (`AccStack`), the forward pass for any depth (residual layers too), and a batched forward pass split across threads (`forward_batch`, `child_logits`) |
| `src/rules.rs` | the game rules, ported from `chaturaji/` (which stays the reference): move generation in the same order as Python, make/unmake (`push`/`pop`), scoring and check bonuses, king captures and eliminations, resignations/timeouts (`eliminate`), turn order, last-king and threefold-repetition endings with their bonuses, board-only Zobrist hashes for repetition, and the network's 16 global features |
| `src/lib.rs` | Python bindings: `Nnue(ft_w, ft_b, layer_w, layer_b, residual, out_w, out_b)` with `.accumulator(board_codes)`, `.refresh`, `.forward`; `Accumulator` with `.push`, `.pop`, `.acc()`, `.logits(k, globals)`, `.child_logits(boards, ks, globals)`; `Position` with `.initial()`, `.from_parts(...)`, `.legal_moves()`, `.push(move)`, `.pop()`, `.eliminate(seat)`, `.globals(k)`, `.board`, `.scores`, `.turn`, `.game_over`, ... |

`bot/rules_core.py` converts between the two engines (`position_from_state`, `move_code`,
`move_from_code`) and checks them against each other:

```powershell
python -m bot.rules_core check     # every scraped game + 200 random games, every ply compared
python -m bot.rules_core bench     # random playouts, Python vs compiled
```

Moves are integers `from_square * 64 + to_square` (square = row * 8 + col, row 0 = rank 8).
Promotion is forced, so it isn't encoded.

Next: the max^n search loop in Rust, on top of `Position` and the accumulators, so a whole
search runs in compiled code.

## Build on Windows

Needs Rust (rustup, MSVC toolchain) and maturin (`pip install maturin`).

```powershell
cd C:\Users\Arnav\Downloads\Chaturaj\engine
python -m maturin build --release
python -m pip install --user --no-deps --force-reinstall (Get-ChildItem target\wheels\*.whl | Select-Object -First 1).FullName
```

Rebuild after changing the Rust code. Check:

```powershell
python -c "import bot.nnue as n; print(n.CORE_AVAILABLE)"
```

`cargo test --no-default-features --release` runs the Rust-only tests (no Python needed);
`python -m pytest tests/test_nnue.py` checks the core against NumPy.

## Build on Tillicum

```bash
bash cluster/build_engine.sh      # installs Rust into /gpfs/scrubbed/$USER if needed, builds, installs into the env
```

## Speed (this laptop, 16 threads)

Rules, random playouts (list the legal moves, make one): 7.7 µs a ply compiled, called from
Python one move at a time (mostly call overhead), vs 264 µs in Python, 34x faster.

NNUE:

Per position, batched evaluation of 14 sibling positions (what max^n does at the leaves):

| Preset | Weights | NumPy | Rust core |
|---|---:|---:|---:|
| small | 0.44 M | 188 µs | 116 µs (mostly Python feature building) |
| medium | 3.5 M | 1,120 µs | 237 µs |
| large | 12.2 M | 3,500 µs | 556 µs |
| xl | 63.6 M | 6,380 µs | 1,540 µs |

Large networks are limited by memory bandwidth: one position reads every weight once
(255 MB for xl), so evaluating siblings together, reading each weight once for the whole
batch, is what makes them usable. Thread count follows `RAYON_NUM_THREADS` (default: all
cores). Set it to 1 when several matches run side by side.

The build targets x86-64-v3 (AVX2 + FMA), set in `.cargo/config.toml`.

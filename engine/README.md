# engine — compiled core (Rust + PyO3)

The Python module `chaturaji_engine`. `bot/nnue.py` uses it automatically when it is
installed (`bot.nnue.CORE_AVAILABLE`), and falls back to NumPy when it isn't. Results are
the same either way; the tests check that.

## What's in it

| File | Contents |
|---|---|
| `src/nnue.rs` | NNUE inference, the same network as `bot/nnue.py`: the feature table, accumulators from scratch (`refresh`), incremental updates from a board difference (`apply_diff`), a push/pop stack for search (`AccStack`), the forward pass for any depth (residual layers too), and a batched forward pass split across threads (`forward_batch`, `child_logits`) |
| `src/lib.rs` | Python bindings: `Nnue(ft_w, ft_b, layer_w, layer_b, residual, out_w, out_b)` with `.accumulator(board_codes)`, `.refresh`, `.forward`; `Accumulator` with `.push`, `.pop`, `.acc()`, `.logits(k, globals)`, `.child_logits(boards, ks, globals)` |

Next: move generation and make/unmake in Rust (checked against the Python rules engine on
all scraped games), then the max^n search itself.

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

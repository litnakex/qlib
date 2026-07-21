# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What this is

Qlib (package name `pyqlib`) is a quantitative-investment research platform. It covers the full quant pipeline: data ingestion, feature/factor computation, model training, portfolio strategy, backtest execution, and analysis. Components are intentionally loose-coupled — each can be used stand-alone.

## Setup & build

The package ships two Cython extensions (`qlib/data/_libs/rolling.pyx`, `expanding.pyx`) that MUST be compiled before the package works. `pip install -e .` alone is NOT enough — the `prerequisite` make target compiles them in-place.

```bash
make install     # prerequisite (compile Cython) + editable install
make dev         # prerequisite + install ALL extras (dev/lint/docs/test/rl/...) — use this for development
make prerequisite  # just compile the .so files (run after cleaning them)
```

Optional dependency groups are defined in `pyproject.toml` under `[project.optional-dependencies]`: `dev`, `rl`, `lint`, `docs`, `test`, `analysis`, `client`. Install individually with `pip install -e .[<group>]`.

Note: `rl` extra pins `numpy<2.0.0` and installs `torch` + `tianshou<=0.4.10`. Torch is NOT a core dependency; deep-learning models require it separately.

## Data

Most tests and examples need downloaded market data (not in the repo):

```bash
python scripts/get_data.py qlib_data --name qlib_data_simple --target_dir ~/.qlib/qlib_data/cn_data --interval 1d --region cn
```

Data lives under `~/.qlib/qlib_data/` by default (the `provider_uri`). `qlib.init(provider_uri=..., region=REG_CN)` must be called before using any data/model API.

## Testing

```bash
cd tests && python -m pytest . -m "not slow"    # CI runs this; excludes slow tests
cd tests && python -m pytest test_pit.py         # single file
cd tests && python -m pytest test_all_pipeline.py::TestAllFlow::test_0_train -v   # single test
```

- Tests must be run from the `tests/` directory (config in `tests/pytest.ini`, shared fixtures in `tests/conftest.py`).
- The `slow` marker gates long-running tests; CI runs `-m "not slow"` with up to 3 retries.
- On macOS, CI sets `OMP_NUM_THREADS=1` (and MKL/NUMEXPR/OPENBLAS/VECLIB) to avoid OpenMP segfaults — do the same if you hit segfaults locally.

## Linting

`make lint` runs all of: `black` (line length 120, excludes `qlib/_version.py`), `pylint`, `flake8`, `mypy`, `nbqa`. Run them individually via `make black` / `make pylint` / `make flake8` / `make mypy` / `make nbqa`. The exact disable-lists live in the `Makefile` — match them, don't invent your own. `mypy` is pinned `<1.5.0`.

Commit messages are checked by commitlint (`.commitlintrc.js`) — use conventional-commit prefixes (`fix:`, `docs:`, `feat:`, etc.), as seen in git history.

## Running workflows

Two equivalent interfaces do the same thing:

1. **Config-driven**: `qrun <config.yaml>` (entry point `qlib.cli.run:run`). Configs live in `examples/benchmarks/*/workflow_config_*.yaml`. YAML is Jinja2-templated against environment variables.
2. **Code-driven**: build blocks manually — see `examples/workflow_by_code.py` for the canonical end-to-end example.

## Architecture

The workflow flows: **data → dataset/handler → model → signal → strategy → executor (backtest) → recorder → analysis**.

**Config as glue** — Almost everything is instantiated from nested dicts of `{"class", "module_path", "kwargs"}` via `qlib.utils.init_instance_by_config`. When adding a new model/strategy/handler, it just needs to be importable by `module_path` + `class`; YAML configs and code both use this same mechanism. This is why components are swappable without touching framework code.

**Key module boundaries:**

- `qlib/data/` — Data provider layer. `data.py` (D provider API), `ops.py` (expression operators for factor computation, backed by the Cython `_libs`), `cache.py` (H memory cache — `qlib.init` clears it), `storage/` (on-disk `.bin` format), `dataset/` (see below), `pit.py` (point-in-time / no-lookahead data).
- `qlib/data/dataset/` — `handler.py` (`DataHandler`/`DataHandlerLP` — loads raw features and applies processors), `loader.py` (raw data loading), `processor.py` (normalization, fillna, etc.), and `DatasetH` which pairs a handler with train/valid/test time segments. `qlib/contrib/data/handler.py` has the common `Alpha158`/`Alpha360` feature sets.
- `qlib/model/` — `base.py` defines the `Model` ABC (`fit(dataset)` / `predict(dataset, segment)`). Concrete models live in `qlib/contrib/model/`. Also `trainer.py` (`task_train`, the unit `qrun` executes), `riskmodel/`, `meta/` (meta-learning), `ens/` (ensembling), `interpret/`.
- `qlib/strategy/base.py` + `qlib/contrib/strategy/` — trading strategies turn model signals into trade decisions (e.g. `TopkDropoutStrategy`).
- `qlib/backtest/` — execution engine. `exchange.py` (trade cost/limit modeling), `executor.py` (`SimulatorExecutor`, `NestedExecutor` for multi-level), `account.py`/`position.py` (portfolio state), `decision.py` (orders). Supports nesting strategies/executors at different frequencies.
- `qlib/workflow/` — experiment management. `R` (the recorder singleton in `qlib/workflow/__init__.py`) wraps MLflow; `R.start()`, `R.log_params`, `R.save_objects`. `record_temp.py` defines `SignalRecord` / `SigAnaRecord` / `PortAnaRecord` — the reusable steps that generate predictions, signal analysis, and portfolio analysis. `task/` and `online/` handle rolling/online serving.
- `qlib/rl/` — reinforcement-learning framework (order execution etc.); `simulator.py`, `interpreter.py`, `reward.py`, `order_execution/`. Needs the `rl` extra.
- `qlib/contrib/` — the "batteries": ready-to-use models, strategies, data handlers, evaluation, reporting, rolling, and tuner utilities. New user-facing implementations usually go here, not in the core packages above.
- `qlib/config.py` — global config `C`. `default_conf` is `client` or `server`; `qlib.init` / `qlib.auto_init` set it up (see `qlib/__init__.py`).

**Scripts** (`scripts/`) are for data prep, not the library: `get_data.py` (download), `dump_bin.py` (CSV → qlib `.bin`), `dump_pit.py`, `check_data_health.py`, `data_collector/`.

## Conventions worth knowing

- New models/strategies/handlers belong in `qlib/contrib/` and must follow the `init_instance_by_config` contract (importable class + kwargs).
- Serialization uses a `RestrictedUnpickler` (`qlib/utils/`) for security — recent CVEs were fixed here; don't reintroduce plain `pickle.load` on untrusted input.
- `pandas>=1.1` is required (fillna method behavior); avoid the deprecated `method=` arg to `fillna`.

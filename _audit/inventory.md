# Inventory — sector-portfolio-optimization

## Tracked files (25)
- 4 notebooks: `01_Data_Collection_and_Cleaning.ipynb`, `02_Exploratory_Data_Analysis.ipynb`, `03_Portfolio_Optimization.ipynb`, `04_Clustering_Optimization.ipynb`
- Python package `quant_sector_optimizer/`: `__init__.py`, `data.py`, `covariance.py`, `optimizers.py`, `backtest.py`, `metrics.py`, `clustering.py`, `plotting.py`
- `tests/` (7 files): `conftest.py` + 6 `test_*.py` — 44 tests collected
- Data: `cleaned_data.parquet` (~33 MB, committed)
- Meta: `README.md`, `LICENSE` (MIT, Copyright 2026 Dan Allouche), `requirements.txt`, `.gitignore`
- New: `_audit/` (Phase 0)

## Untracked on disk
- `.venv/` (project venv, ignored)
- `.venv-audit/` (audit-only venv created by sub-agent)
- `.pytest_cache/`, `quant_sector_optimizer/__pycache__/`, `tests/__pycache__/`
- `_audit/abs_paths.txt` (empty, 0 lines), `_audit/secrets_trufflehog.txt` (339 raw hits, all `.venv/` httpx vendored examples, verified=0)

## Identity
- README: `**Author:** Dan Allouche` — already normalized (Cas B nominal, no action)
- LICENSE: `Copyright (c) 2026 Dan Allouche`
- Notebook metadata: `authors=None` everywhere (no field present)
- Git log authors: `Dan Allouche <dan.allouche@icloud.com>` (clean)
- No "Joseph", no "D. Allouche", no AI mentions in source

## Subject classification
PUBLIC_QUANT — quant portfolio toolkit (mean-variance, max-Sharpe convex QP, max-div, HRP, Ledoit-Wolf shrinkage, clustering). Strong package + 4 demo notebooks + tests + license + pinned deps. Tier A candidate.

## Phase 0 inputs (already done)
- Branch `audit-2026-05` active, tag `pre-audit-2026-05/main = 695f9c6`, pre-push hook installed (exit 1)
- Scans: 0 abs-paths, 339 raw secrets hits — ALL false positives (httpx `example.com` URLs in vendored `.venv`)

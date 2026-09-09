# Paper materials

`manuscript/main.tex`, `manuscript/refe.bib`, and Figures 1–5 are the canonical public manuscript sources dated 2026-03-23.

`published/Agri-MetaRL-2026-03-19.pdf` is the latest standalone PDF found during reorganization. It predates the canonical manuscript source and therefore may not reproduce that revision exactly.

Compile from `paper/manuscript/` so the bibliography remains local and figure references resolve through `../figures/`:

```bash
latexmk -pdf -outdir=../../artifacts/paper-build main.tex
```

Build output belongs under `artifacts/` and is not tracked.

## C-route manuscript source policy

The redesigned manuscript must use `artifacts/results/AgriControl_C_2026-06-30/` as its result source.

Do not cite summaries, figures, or raw rows from `artifacts/results/AgriControl/` for the C-route manuscript. That directory is archival and contains stale or incomplete experiment outputs.

Before updating tables or figures, run:

```powershell
python experiments\scripts\validate_suite_artifacts.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json
python experiments\scripts\summarize_suite.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json
python experiments\scripts\generate_suite_figures.py --manifest artifacts\results\AgriControl_C_2026-06-30\suite_manifest.json
```

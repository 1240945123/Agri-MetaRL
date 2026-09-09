# Agri-MetaRL Repository Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize Agri-MetaRL into a clean academic-engineering repository without deleting any existing file or losing the current uncommitted table-formatting change.

**Architecture:** Perform a content-preserving physical migration first, verify every moved file by SHA-256, and only then update package paths, experiment commands, documentation, and metadata. Centralize repository paths in `src/gl_gym/paths.py`; keep public code and paper materials tracked while local datasets, artifacts, and private archives remain ignored.

**Tech Stack:** Python 3.11, setuptools/`pyproject.toml`, pathlib, pytest/unittest, PowerShell, Git, LaTeX static checks.

**Commit policy:** Do not create commits automatically. Each task ends with a Git review checkpoint instead of a commit.

---

## File Structure Map

**Create**

- `src/gl_gym/paths.py`: single source of truth for repository, config, dataset, artifact, model, result, and figure paths.
- `tests/test_paths.py`: path-layout contract tests.
- `datasets/README.md`: expected weather dataset layout and local-data policy.
- `paper/README.md`: canonical-source and standalone-PDF provenance.
- `artifacts/.gitkeep`: unnecessary because `artifacts/` is local-only; do not create it if absent after moving outputs.
- `archive/.gitkeep`: unnecessary because `archive/` is local-only; do not track it.
- `.reorganization/pre-move.csv`: ignored temporary inventory created immediately before migration.
- `.reorganization/post-move.csv`: ignored temporary inventory created immediately after migration.

**Move without content changes**

- `gl_gym/` → `src/gl_gym/`, except configs and weather data listed below.
- `gl_gym/configs/` → `configs/`.
- `gl_gym/environments/weather/` → `datasets/weather/`.
- `run_scripts/` → `experiments/scripts/`.
- `train_data/` → `artifacts/models/`.
- `data/` → `artifacts/results/`.
- `wandb/` → `artifacts/tracking/`.
- `plots/`, `visualisations/`, and noncanonical generated visual files → `artifacts/figures/` or `archive/previous_versions/` according to whether they are generated output or old source.
- Latest public manuscript files from `tougao/` → `paper/manuscript/` and `paper/figures/`.
- `argi-meta-rl-f.pdf` → `paper/published/Agri-MetaRL-2026-03-19.pdf`.
- Older manuscripts, LaTeX outputs, obsolete figures, and reference PDFs → `archive/previous_versions/`.
- Submission forms, reports, cover letters, `tougao/Agri_Meta.zip`, and declarations → `archive/submission_private/`.
- `签名/`, author agreements, and signature-related images → `archive/signatures/`.

**Modify after move verification**

- `pyproject.toml`, `.gitignore`, `README.md`, `CITATION.cff`.
- `src/gl_gym/common/utils.py`.
- `src/gl_gym/RL/experiment_manager.py`.
- `src/gl_gym/experiments/{stochastic_rl.py,run_time.py,gl_predefined_controls.py,evaluate_rl.py,evaluate_baseline.py}`.
- `configs/envs/TomatoEnv.yml`, `configs/README.md`, `src/gl_gym/environments/README.md`.
- Every Python, batch, shell, and Markdown file under `experiments/scripts/` that references an old path.
- `tests/env_test.py`, root `test_env.py`, and root `test_fixes.py` as required by the new import/config layout.
- `paper/manuscript/main.tex` only where figure or bibliography paths need adjustment.

### Task 1: Capture the Safety Baseline

**Files:**
- Create: `.reorganization/pre-move.csv`
- Inspect: `run_scripts/update_greenlight_table.py`

- [ ] **Step 1: Confirm the repository root and dirty state**

Run:

```powershell
$root = (Resolve-Path 'E:\school\Paper\new').Path
if ($root -ne 'E:\school\Paper\new') { throw "Unexpected repository root: $root" }
git status --short --branch
```

Expected: branch `main`; the existing modification to `run_scripts/update_greenlight_table.py`; existing untracked paper/submission files; the new design and plan documents.

- [ ] **Step 2: Record the EPI change before moving anything**

Run:

```powershell
git diff -- run_scripts/update_greenlight_table.py
```

Expected: additions for `epi_std`, `epi_std_t`, and formatted `\pm` output remain visible.

- [ ] **Step 3: Create a full pre-move content inventory outside Git visibility**

Run:

```powershell
New-Item -ItemType Directory -Force -Path '.reorganization' | Out-Null
Get-ChildItem -Recurse -File -Force |
  Where-Object { $_.FullName -notmatch '\\.git\\|\\.reorganization\\' } |
  ForEach-Object {
    [PSCustomObject]@{
      RelativePath = [IO.Path]::GetRelativePath($root, $_.FullName)
      Length = $_.Length
      LastWriteTimeUtc = $_.LastWriteTimeUtc.ToString('o')
      SHA256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
    }
  } | Sort-Object RelativePath | Export-Csv '.reorganization\pre-move.csv' -NoTypeInformation -Encoding UTF8
```

Expected: `.reorganization/pre-move.csv` exists and has one row per non-Git file.

- [ ] **Step 4: Record baseline totals**

Run:

```powershell
$pre = Import-Csv '.reorganization\pre-move.csv'
$pre.Count
($pre | Measure-Object Length -Sum).Sum
```

Expected: count is greater than 800 and total bytes are greater than 1 GB.

- [ ] **Step 5: Review checkpoint**

Run `git status --short`. Expected: no project files have moved or changed during this task.

### Task 2: Move the Repository Without Editing Content

**Files:**
- Move: all paths in the File Structure Map
- Create: `.reorganization/move-map.csv`

- [ ] **Step 1: Create and validate destination directories**

Run a PowerShell block that resolves every destination under `$root`, rejects any destination outside `$root`, and then creates only these directories:

```powershell
$destinations = @(
  'src', 'experiments', 'datasets',
  'artifacts', 'artifacts\figures',
  'paper\manuscript', 'paper\figures', 'paper\published',
  'archive\previous_versions', 'archive\submission_private', 'archive\signatures'
)
foreach ($relative in $destinations) {
  $target = [IO.Path]::GetFullPath((Join-Path $root $relative))
  if (-not $target.StartsWith($root + [IO.Path]::DirectorySeparatorChar)) { throw "Unsafe target: $target" }
  New-Item -ItemType Directory -Force -Path $target | Out-Null
}
```

Expected: every destination resolves beneath `E:\school\Paper\new`.

- [ ] **Step 2: Move source, configuration, scripts, datasets, and artifacts**

Before each `Move-Item`, require `Test-Path` for the source and require the final destination not to exist. Use `Move-Item -LiteralPath` within PowerShell only; do not invoke another shell. Execute these mappings in order:

```text
gl_gym/configs                         -> configs
gl_gym/environments/weather            -> datasets/weather
gl_gym                                 -> src/gl_gym
run_scripts                            -> experiments/scripts
train_data                             -> artifacts/models
data                                   -> artifacts/results
wandb                                  -> artifacts/tracking
plots                                  -> artifacts/figures/legacy-plots
visualisations                         -> artifacts/figures/generators
processing                             -> archive/previous_versions/processing
images                                 -> archive/previous_versions/images
visual                                 -> archive/previous_versions/visual
```

Expected: sources no longer exist at their old paths and every target exists.

- [ ] **Step 3: Move canonical public paper materials**

Use collision-safe `Move-Item -LiteralPath` calls for:

```text
tougao/main.tex                         -> paper/manuscript/main.tex
tougao/refe.bib                         -> paper/manuscript/refe.bib
tougao/main.bbl                         -> paper/manuscript/main.bbl
tougao/Figure_1.pdf ... Figure_5.pdf    -> paper/figures/Figure_1.pdf ... Figure_5.pdf
argi-meta-rl-f.pdf                      -> paper/published/Agri-MetaRL-2026-03-19.pdf
```

Expected: all eleven canonical files exist in `paper/` and retain their original hashes.

- [ ] **Step 4: Classify private and legacy files without deleting anything**

Move submission/private material into `archive/submission_private/`, signatures and agreements into `archive/signatures/`, and all remaining older paper/reference/LaTeX material into `archive/previous_versions/`. Preserve original filenames unless a collision occurs; for collisions, append the original parent directory or `yyyy-MM-dd` modification date. Include the remaining contents of `tougao/` and remove the empty `tougao/` directory only after confirming it contains no files.

Expected: root contains no loose PDF, DOCX, PPTX, TEX, BIB, AUX, BBL, BLG, LOG, OUT, or SPL files except project metadata explicitly intended to remain there.

- [ ] **Step 5: Save the exact old-to-new move map**

Write `.reorganization/move-map.csv` with columns `OriginalPath`, `NewPath`, `Length`, and `SHA256`. Populate it from the operations actually executed, not from the intended mapping.

- [ ] **Step 6: Review checkpoint**

Run:

```powershell
git status --short
```

Expected: many deletes/adds or renames are visible, but no commit exists and the EPI-modified script is now under `experiments/scripts/`.

### Task 3: Prove the Physical Migration Preserved Content

**Files:**
- Create: `.reorganization/post-move.csv`
- Read: `.reorganization/pre-move.csv`, `.reorganization/move-map.csv`

- [ ] **Step 1: Generate the post-move inventory before editing any migrated file**

Use the same inventory command as Task 1 and export to `.reorganization/post-move.csv`.

- [ ] **Step 2: Verify every moved hash**

For every row in `move-map.csv`, hash `NewPath` and assert equality with `SHA256`. Throw immediately on a missing path or mismatch.

Expected: zero missing files and zero hash mismatches.

- [ ] **Step 3: Account for every pre-move file**

Compare unchanged paths directly and moved paths through `move-map.csv`. Exclude only the newly generated `.reorganization` inventories. Print separate lists for missing, hash-mismatched, and unaccounted files.

Expected:

```text
Missing: 0
HashMismatch: 0
Unaccounted: 0
```

- [ ] **Step 4: Stop gate**

Do not begin code or documentation edits unless all three counts are zero. If any count is nonzero, leave all files in their current locations and report the exact paths for review.

### Task 4: Add a Tested Repository Path Contract

**Files:**
- Create: `src/gl_gym/paths.py`
- Create: `tests/test_paths.py`

- [ ] **Step 1: Write failing path-layout tests**

Create tests asserting these exact contracts:

```python
from gl_gym.paths import (
    PROJECT_ROOT, CONFIG_DIR, WEATHER_DIR, ARTIFACT_DIR,
    MODEL_DIR, RESULT_DIR, GENERATED_FIGURE_DIR,
)


def test_project_paths_match_repository_layout():
    assert PROJECT_ROOT.name == "new"
    assert CONFIG_DIR == PROJECT_ROOT / "configs"
    assert WEATHER_DIR == PROJECT_ROOT / "datasets" / "weather"
    assert ARTIFACT_DIR == PROJECT_ROOT / "artifacts"
    assert MODEL_DIR == ARTIFACT_DIR / "models"
    assert RESULT_DIR == ARTIFACT_DIR / "results"
    assert GENERATED_FIGURE_DIR == ARTIFACT_DIR / "figures"


def test_required_local_directories_exist():
    assert CONFIG_DIR.is_dir()
    assert WEATHER_DIR.is_dir()
    assert MODEL_DIR.is_dir()
    assert RESULT_DIR.is_dir()
```

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```powershell
$env:PYTHONPATH = (Join-Path $root 'src')
python -m pytest tests/test_paths.py -v
```

Expected: FAIL because `gl_gym.paths` does not exist.

- [ ] **Step 3: Implement the minimal path module**

Create:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATASET_DIR = PROJECT_ROOT / "datasets"
WEATHER_DIR = DATASET_DIR / "weather"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
RESULT_DIR = ARTIFACT_DIR / "results"
GENERATED_FIGURE_DIR = ARTIFACT_DIR / "figures"
TRACKING_DIR = ARTIFACT_DIR / "tracking"
PAPER_DIR = PROJECT_ROOT / "paper"
```

- [ ] **Step 4: Run the focused test and verify success**

Run `python -m pytest tests/test_paths.py -v` with `PYTHONPATH=src`.

Expected: 2 passed.

- [ ] **Step 5: Review checkpoint**

Run `git diff -- src/gl_gym/paths.py tests/test_paths.py`. Do not commit.

### Task 5: Convert Packaging, Config, and Library Code to the New Layout

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/gl_gym/common/utils.py`
- Modify: `src/gl_gym/RL/experiment_manager.py`
- Modify: `src/gl_gym/experiments/stochastic_rl.py`
- Modify: `src/gl_gym/experiments/run_time.py`
- Modify: `src/gl_gym/experiments/gl_predefined_controls.py`
- Modify: `src/gl_gym/experiments/evaluate_rl.py`
- Modify: `src/gl_gym/experiments/evaluate_baseline.py`
- Modify: `configs/envs/TomatoEnv.yml`
- Modify: `configs/README.md`
- Modify: `src/gl_gym/environments/README.md`
- Modify: `tests/env_test.py`
- Modify: `test_env.py`

- [ ] **Step 1: Update the setuptools `src` configuration**

Set:

```toml
[tool.setuptools.packages.find]
where = ["src"]
include = ["gl_gym*"]
```

Keep `setup.py` only if editable installation still requires it in the existing environment; otherwise remove it only after confirming it is tracked and redundant under PEP 517. A removed tracked file remains recoverable in Git history and must be explicitly shown in the review checkpoint.

- [ ] **Step 2: Replace library hard-coded roots with `gl_gym.paths` constants**

Use `CONFIG_DIR / "agents"`, `CONFIG_DIR / "envs"`, `CONFIG_DIR / "sweeps"`, `MODEL_DIR`, and `RESULT_DIR` in the listed library files. Convert to `str(...)` only at APIs that require strings. Preserve the current algorithm behavior and filenames.

- [ ] **Step 3: Update the environment YAML**

Set the weather path to:

```yaml
weather_data_dir: datasets/weather
```

Do not embed an absolute machine-specific path.

- [ ] **Step 4: Update config-loading tests before implementation checks**

Replace test literals such as `gl_gym/configs/envs/` with `str(CONFIG_DIR / "envs")`. Add an assertion that the TomatoEnv YAML resolves its weather directory to `WEATHER_DIR` when invoked from the repository root.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
$env:PYTHONPATH = (Join-Path $root 'src')
python -m pytest tests/test_paths.py tests/env_test.py -v
```

Expected: imports succeed and tests pass; if the long episode-termination test is slow, run the remaining named tests first and report it separately rather than silently skipping it.

- [ ] **Step 6: Validate package discovery without installation**

Run:

```powershell
python -c "import sys; sys.path.insert(0, 'src'); import gl_gym; print(gl_gym.__file__)"
```

Expected: printed path begins with `E:\school\Paper\new\src\gl_gym`.

- [ ] **Step 7: Review checkpoint**

Run `git diff -- pyproject.toml src/gl_gym configs tests test_env.py`. Do not commit.

### Task 6: Update Every Experiment Entry Point

**Files:**
- Modify: all `*.py`, `*.bat`, `*.sh`, and `*.md` under `experiments/scripts/`

- [ ] **Step 1: Establish canonical defaults**

Use these defaults consistently:

```text
configs/envs                         -> environment configs
artifacts/models/AgriControl         -> trained models
artifacts/models/AgriControl_ablation -> ablation models
artifacts/results/AgriControl        -> evaluation CSV outputs
artifacts/figures                    -> generated figures
experiments/scripts                  -> subprocess entry points
```

- [ ] **Step 2: Update Python scripts**

Replace old `run_scripts`, `train_data`, `data/AgriControl`, `gl_gym/configs`, `visualisations`, and `visual` references. Import path constants from `gl_gym.paths`. In `run_paper_pipeline_after_train.py`, construct subprocess commands with `sys.executable` and absolute `Path` objects under `PROJECT_ROOT / "experiments" / "scripts"` instead of shell command strings.

- [ ] **Step 3: Update batch and shell scripts**

Change command paths to `experiments\scripts\...` on Windows and `experiments/scripts/...` in shell. Change printed output locations to `artifacts/...`. Do not add compatibility calls to the old directories.

- [ ] **Step 4: Preserve the table-formatting change**

Inspect `experiments/scripts/update_greenlight_table.py` and confirm the relocated file still contains:

```python
epi_std = fixed.loc[method, ("EPI", "std")] if ("EPI", "std") in fixed.columns else 0
epi_std_t = heldout.loc[method, ("EPI", "std")] if ("EPI", "std") in heldout.columns else 0
```

Retain the `\pm` formatting for positive EPI values while updating only input/output paths.

- [ ] **Step 5: Prove no old active path remains**

Run:

```powershell
rg -n --glob '*.py' --glob '*.bat' --glob '*.sh' --glob '*.md' \
  'run_scripts|train_data|gl_gym[/\\]configs|environments[/\\]weather|visualisations|data[/\\]AgriControl' \
  src configs experiments tests README.md test_env.py
```

Expected: no executable/documentation references remain; historical mentions in the approved design/plan are excluded from this search.

- [ ] **Step 6: Exercise non-mutating CLI checks**

Run `python <script> --help` for all Python files under `experiments/scripts/` that define an argument parser. For scripts without `--help`, import them with `runpy` only if their module top level has no side effects; otherwise perform `python -m py_compile` instead.

Expected: zero import, syntax, or path-bootstrap errors and no training starts.

- [ ] **Step 7: Review checkpoint**

Run `git diff -- experiments/scripts`. Do not commit.

### Task 7: Finalize Public Paper and Private Archive Separation

**Files:**
- Create: `paper/README.md`
- Modify: `paper/manuscript/main.tex`
- Inspect: `paper/figures/Figure_1.pdf` through `Figure_5.pdf`
- Inspect: `paper/published/Agri-MetaRL-2026-03-19.pdf`

- [ ] **Step 1: Document canonical provenance**

State in `paper/README.md` that `paper/manuscript/main.tex` and its figures are dated 2026-03-23, while the latest available standalone PDF is dated 2026-03-19 and may not exactly reproduce that source revision.

- [ ] **Step 2: Update manuscript paths**

Change figure references to `../figures/Figure_N` only if compilation is run from `paper/manuscript/`; keep bibliography resolution local to `refe.bib`. Use one consistent LaTeX working-directory convention and document the exact command.

- [ ] **Step 3: Statically validate all paper references**

Run `rg` against `paper/manuscript/main.tex` to enumerate `\includegraphics` and bibliography commands. Assert Figure 1-5 files and `refe.bib` exist at the resolved paths.

Expected: no missing figure or bibliography file.

- [ ] **Step 4: Compile only if tooling already exists**

Run `Get-Command latexmk -ErrorAction SilentlyContinue`. If present, compile from `paper/manuscript/` with output directed to `artifacts/paper-build/`. If absent, report static validation only; do not install LaTeX.

- [ ] **Step 5: Verify archive privacy**

List all files under `archive/signatures/` and `archive/submission_private/` for local accounting, but do not print image/PDF contents. Confirm Git ignore rules from Task 8 will hide both directories.

- [ ] **Step 6: Review checkpoint**

Run `git status --short -- paper archive`. Expected at this stage: public `paper/` files and untracked `archive/` files are visible; Task 8 will add the ignore rule that hides `archive/`.

### Task 8: Rewrite Ignore Rules, Metadata, and Documentation

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`
- Modify: `pyproject.toml`
- Modify: `CITATION.cff`
- Create: `datasets/README.md`
- Modify: `experiments/scripts/README_scripts.md`

- [ ] **Step 1: Replace broad ignore patterns**

The new `.gitignore` must ignore:

```gitignore
.vscode/
.idea/
__pycache__/
*.py[cod]
build/
dist/
*.egg-info/
.reorganization/
archive/
artifacts/
datasets/weather/
wandb/
*.aux
*.log
*.out
*.blg
*.spl
```

Do not ignore all `*.tex`, `*.bib`, PDFs, `paper/`, or `datasets/README.md`.

- [ ] **Step 2: Rewrite the root README in valid UTF-8**

Include the new structure, `src` installation, canonical `experiments/scripts/` commands, local weather-data placement, artifact outputs, public paper links, and the canonical-source/PDF date distinction. Remove corrupted characters and links to nonexistent `visual/Figure_2.pdf` through `Figure_5.pdf`.

- [ ] **Step 3: Correct project metadata**

Set `pyproject.toml` name/description/URLs/authors to the Agri-MetaRL project using the author names already present in the manuscript and README. Update `CITATION.cff` from the upstream GreenLight-Gym citation to the Agri-MetaRL paper citation, keeping valid CFF 1.2.0 structure.

- [ ] **Step 4: Document local datasets**

In `datasets/README.md`, list the expected `datasets/weather/<location>/<year>.csv` layout, explain that CSV files are intentionally not tracked because they total roughly 200 MB, and state that existing local files were preserved during reorganization. Do not invent a download URL if the repository does not contain an authoritative source.

- [ ] **Step 5: Check text encoding and links**

Read README files explicitly as UTF-8 and search for replacement characters or known mojibake fragments. Check every relative Markdown link in the root README resolves to an existing path.

- [ ] **Step 6: Check Git visibility policy**

Run:

```powershell
git check-ignore -v archive/signatures/* archive/submission_private/* artifacts/models/* datasets/weather/*
git check-ignore -v paper/manuscript/main.tex paper/figures/Figure_1.pdf
```

Expected: the first group is ignored; the public paper files are not ignored.

- [ ] **Step 7: Review checkpoint**

Run `git diff -- .gitignore README.md pyproject.toml CITATION.cff datasets paper experiments/scripts/README_scripts.md`. Do not commit.

### Task 9: Final Verification and Handoff

**Files:**
- Read: all changed files
- Preserve: `.reorganization/*.csv` as ignored local audit artifacts

- [ ] **Step 1: Run the full lightweight test set**

Run:

```powershell
$env:PYTHONPATH = (Join-Path $root 'src')
python -m pytest -q
```

Expected: all collected tests pass. Report any pre-existing failure separately with its traceback; do not claim completion while new path-related failures remain.

- [ ] **Step 2: Compile Python sources**

Run:

```powershell
python -m compileall -q src experiments/scripts tests test_env.py test_fixes.py
```

Expected: exit code 0.

- [ ] **Step 3: Re-run stale-path search**

Search active code and docs for every old root. Expected: zero active references, excluding historical design/plan documents and ignored inventories.

- [ ] **Step 4: Verify final repository shape**

Confirm the root contains the intended public directories and no loose submission files. Report directory file counts and sizes for `src`, `configs`, `experiments`, `datasets`, `artifacts`, `paper`, `archive`, `docs`, and `tests`.

- [ ] **Step 5: Verify final Git state**

Run:

```powershell
git status --short --branch
git diff --stat
git diff -- experiments/scripts/update_greenlight_table.py
```

Expected: intentional tracked moves and edits only; archive/artifact/weather contents hidden; EPI standard-deviation behavior retained; no commit created.

- [ ] **Step 6: Report completion with evidence**

Provide test counts, CLI-check counts, hash-accounting totals, paper static/compile status, final Git summary, and any intentionally retained caveats. Do not stage or commit files.

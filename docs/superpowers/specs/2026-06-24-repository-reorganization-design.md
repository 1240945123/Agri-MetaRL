# Agri-MetaRL Repository Reorganization Design

## Objective

Reorganize the repository into a complete academic-engineering structure while preserving every existing file. The result must separate source code, configuration, experiments, datasets, generated artifacts, public paper materials, and private archives. Existing command paths may change; documentation and code will be updated to use the new structure.

## Constraints

- Do not delete existing files.
- Do not install dependencies, runtimes, caches, or temporary projects on the C: drive.
- Keep all project changes within `E:\school\Paper\new`.
- Do not rewrite Git history.
- Do not commit changes automatically.
- Preserve the uncommitted EPI standard-deviation changes in `run_scripts/update_greenlight_table.py`.
- Do not run long training jobs during verification.

## Target Structure

```text
new/
|-- src/gl_gym/                 # Python package source
|   |-- common/
|   |-- environments/
|   |-- experiments/
|   `-- RL/
|-- configs/                    # Agent, environment, and sweep configuration
|-- experiments/
|   `-- scripts/                # Training, evaluation, and plotting entry points
|-- datasets/
|   `-- weather/                # Greenhouse weather input data
|-- artifacts/                  # Regenerable local outputs; ignored by Git
|   |-- models/                 # Former train_data
|   |-- results/                # Former data evaluation outputs
|   |-- figures/                # Generated experiment figures
|   `-- tracking/               # Former wandb data
|-- paper/                      # Public paper materials
|   |-- manuscript/             # Latest manuscript source and bibliography
|   |-- figures/                # Latest Figures 1-5
|   `-- published/              # Latest available standalone paper PDF
|-- archive/                    # Local archive; ignored by Git
|   |-- previous_versions/      # Older manuscripts, figures, and LaTeX outputs
|   |-- submission_private/     # Submission forms, reports, and packages
|   `-- signatures/             # Signatures and author agreements
|-- docs/
|-- tests/
|-- README.md
|-- pyproject.toml
|-- CITATION.cff
|-- LICENSE
`-- .gitignore
```

## Migration Map

| Current location | Target location | Git policy |
|---|---|---|
| `gl_gym/` Python source | `src/gl_gym/` | Track |
| `gl_gym/configs/` | `configs/` | Track |
| `gl_gym/environments/weather/` | `datasets/weather/` | Preserve locally and ignore; track only a data README/manifest |
| `run_scripts/` | `experiments/scripts/` | Track |
| `train_data/` | `artifacts/models/` | Ignore |
| `data/` | `artifacts/results/` | Ignore |
| `wandb/` | `artifacts/tracking/` | Ignore |
| Generated plots and figures not selected for the paper | `artifacts/figures/` | Ignore |
| Latest `tougao/main.tex`, `refe.bib`, and `main.bbl` | `paper/manuscript/` | Track |
| Latest `tougao/Figure_1.pdf` through `Figure_5.pdf` | `paper/figures/` | Track |
| `argi-meta-rl-f.pdf` | `paper/published/` | Track, with provenance documented |
| Older paper versions and LaTeX build outputs | `archive/previous_versions/` | Ignore |
| Submission packages, cover letters, declarations, and reports | `archive/submission_private/` | Ignore |
| Signature images and author agreements | `archive/signatures/` | Ignore |

The manuscript source dated 2026-03-23 is the canonical public source. The standalone PDF dated 2026-03-19 is the latest existing PDF but predates that source; the README must state this rather than imply an exact source-to-PDF match.

## Path and Packaging Changes

- Convert `pyproject.toml` to a standard `src` package layout.
- Update package discovery to find `gl_gym` under `src/`.
- Update experiment scripts, batch files, tests, and package code to resolve the new `configs/`, `datasets/`, and `artifacts/` paths from the repository root.
- Replace scattered hard-coded output paths with shared path helpers where needed.
- Do not retain compatibility wrappers for old commands.
- Document only the new commands under `experiments/scripts/`.

## Version-Control Policy

- Track source code, configuration, tests, project documentation, citation metadata, the canonical paper source, public paper figures, and the latest available public PDF.
- Ignore the complete `archive/` and `artifacts/` trees.
- Ignore weather data files under `datasets/weather/`; track a manifest and acquisition/placement instructions so the expected dataset layout is explicit without adding roughly 200 MB of local data to Git.
- Replace broad ignore rules such as `*.tex`, `*.bib`, and `visual/*` with responsibility-based rules that do not hide public paper sources.
- Keep private submission and signature material outside Git visibility after migration.
- Let Git detect tracked moves as renames where possible; do not rewrite history.

## Documentation and Metadata

- Rewrite the README so its directory tree, installation instructions, commands, output locations, and figure links match the new structure.
- Correct visible encoding corruption in README and script documentation.
- Update `pyproject.toml` project identity, author information, and project URLs to Agri-MetaRL while preserving valid dependency constraints.
- Review `CITATION.cff` against the paper metadata and make it consistent with the README citation.
- Document the distinction between the canonical manuscript source and the older standalone PDF.

## Safety and Error Handling

1. Produce a pre-migration inventory containing relative path, byte size, modification time, and SHA-256 hash.
2. Move files without overwriting existing targets.
3. Resolve name collisions by retaining both files with source- or date-qualified names.
4. If a file's destination is ambiguous, leave it in place and record it for review.
5. Produce a post-migration inventory and compare file counts, total bytes, and hashes with the original inventory.
6. Treat a missing or changed file hash as a failed migration and stop before further cleanup.

## Verification

- Confirm the pre- and post-migration inventories contain the same file content.
- Confirm `archive/` and `artifacts/` do not appear as untracked Git content.
- Confirm intended public files remain visible to Git.
- Install nothing during structural verification.
- Verify `gl_gym` imports from the new `src` layout using the existing environment.
- Run the existing test suite and lightweight environment checks.
- Run major experiment entry points with `--help` or an equivalent non-mutating check.
- Statically check manuscript references to Figures 1-5 and the bibliography.
- Compile the manuscript only if a LaTeX toolchain is already available; otherwise report the static-check result.
- Check README links and commands against the final filesystem.
- Confirm the EPI standard-deviation logic remains in the relocated table-update script.

## Completion Criteria

The reorganization is complete when every original file is accounted for, public and private materials are separated, the package and lightweight tests work from the new paths, paper references resolve, documentation matches reality, Git shows only the intentional tracked changes, and no commit has been created automatically.

"""Canonical filesystem locations for the Agri-MetaRL repository."""

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

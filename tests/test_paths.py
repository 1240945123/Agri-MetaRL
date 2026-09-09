from gl_gym.paths import (
    ARTIFACT_DIR,
    CONFIG_DIR,
    GENERATED_FIGURE_DIR,
    MODEL_DIR,
    PROJECT_ROOT,
    RESULT_DIR,
    WEATHER_DIR,
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

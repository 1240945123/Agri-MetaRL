from pathlib import Path


FORBIDDEN = (
    "_LEARNING_SCALE",
    "_ECONOMIC_SCALE",
    "scale rewards so Agri-MetaRL",
    "learning_order=True",
)

ACTIVE_ROOTS = (Path("src"), Path("experiments"))


def find_forbidden_scaling(paths):
    violations = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for marker in FORBIDDEN:
            if marker in text:
                violations.append(f"{path}: {marker}")
    return violations


def active_python_files():
    for root in ACTIVE_ROOTS:
        yield from root.rglob("*.py")


def test_active_code_contains_no_algorithm_dependent_result_scaling():
    assert find_forbidden_scaling(active_python_files()) == []

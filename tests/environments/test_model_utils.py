import pytest

from gl_gym.environments.models import utils as model_utils


def _capture_integrator_call(monkeypatch):
    calls = {}
    integrator = object()

    def fake_ode(x, u, d, p):
        calls["ode_shapes"] = tuple(symbol.numel() for symbol in (x, u, d, p))
        return model_utils.ca.SX.zeros(x.numel())

    def fake_integrator(*args):
        calls.setdefault("integrator_args", []).append(args)
        return integrator

    monkeypatch.setattr(model_utils, "ODE", fake_ode)
    monkeypatch.setattr(model_utils.ca, "integrator", fake_integrator)
    return calls, integrator


def test_define_model_uses_formal_cvodes_defaults_and_requested_horizon(monkeypatch):
    calls, expected_integrator = _capture_integrator_call(monkeypatch)

    actual_integrator = model_utils.define_model(2, 1, 3, 4, 0.25)

    assert actual_integrator is expected_integrator
    assert calls["ode_shapes"] == (2, 1, 3, 4)
    args = calls["integrator_args"][0]
    assert args[:2] == ("F", "cvodes")
    assert args[3:5] == (0.0, 0.25)
    assert args[2]["p"].numel() == 7
    assert args[5] == {
        "abstol": 1e-4,
        "reltol": 1e-4,
        "max_num_steps": 70_000,
    }
    assert dict(model_utils.FORMAL_CVODES_OPTIONS) == args[5]


def test_define_model_merges_diagnostic_overrides(monkeypatch):
    calls, _ = _capture_integrator_call(monkeypatch)
    overrides = {"abstol": 2e-6, "max_order": 3}

    model_utils.define_model(1, 1, 1, 1, 60.0, integrator_options=overrides)

    assert calls["integrator_args"][0][5] == {
        "abstol": 2e-6,
        "reltol": 1e-4,
        "max_num_steps": 70_000,
        "max_order": 3,
    }
    assert overrides == {"abstol": 2e-6, "max_order": 3}


def test_formal_options_are_immutable_and_each_call_gets_a_fresh_copy(monkeypatch):
    calls, _ = _capture_integrator_call(monkeypatch)

    with pytest.raises(TypeError):
        model_utils.FORMAL_CVODES_OPTIONS["abstol"] = 1.0

    model_utils.define_model(1, 1, 1, 1, 1.0)
    first_options = calls["integrator_args"][0][5]
    first_options["abstol"] = 1.0
    model_utils.define_model(1, 1, 1, 1, 1.0)
    second_options = calls["integrator_args"][1][5]

    assert first_options is not second_options
    assert second_options == dict(model_utils.FORMAL_CVODES_OPTIONS)

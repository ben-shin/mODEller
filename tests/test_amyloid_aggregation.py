import pytest

from odefit.api.backend import fit_amyloid_aggregation_from_config
from odefit.api.serialization import backend_output_payload
from odefit.fitting.amyloid_aggregation import (
    AmyloidCondition,
    fit_amyloid_aggregation,
    simulate_amyloid_condition,
)
from odefit.fitting.parameter_spec import ParameterSpec


def _true_parameters():
    return {
        "log_kn": -1.0,
        "log_k2": -1.0,
        "log_ke": 0.0,
        "nc": 1.0,
        "n2": 1.0,
        "cm": 0.05,
        "A1000000": 1.0,
        "A800000": 1.0,
    }


def _parameter_specs():
    return [
        ParameterSpec(
            "log_kn",
            initial_guess=-1.0,
            lower_bound=-3.0,
            upper_bound=3.0,
            fixed=True,
            fixed_value=-1.0,
        ),
        ParameterSpec(
            "log_k2",
            initial_guess=-1.0,
            lower_bound=-3.0,
            upper_bound=3.0,
            fixed=True,
            fixed_value=-1.0,
        ),
        ParameterSpec(
            "log_ke",
            initial_guess=-0.5,
            lower_bound=-3.0,
            upper_bound=3.0,
        ),
        ParameterSpec(
            "nc",
            initial_guess=1.0,
            lower_bound=1.0,
            upper_bound=5.0,
            fixed=True,
            fixed_value=1.0,
        ),
        ParameterSpec(
            "n2",
            initial_guess=1.0,
            lower_bound=1.0,
            upper_bound=4.0,
            fixed=True,
            fixed_value=1.0,
        ),
        ParameterSpec(
            "cm",
            initial_guess=0.05,
            lower_bound=0.001,
            upper_bound=0.2,
            fixed=True,
            fixed_value=0.05,
        ),
        ParameterSpec(
            "A1000000",
            initial_guess=1.0,
            lower_bound=0.5,
            upper_bound=1.5,
            fixed=True,
            fixed_value=1.0,
        ),
        ParameterSpec(
            "A800000",
            initial_guess=1.0,
            lower_bound=0.5,
            upper_bound=1.5,
            fixed=True,
            fixed_value=1.0,
        ),
    ]


def _synthetic_condition(name, mtot, amplitude_parameter):
    condition = AmyloidCondition(
        name=name,
        mtot=mtot,
        timepoints=[0.0, 0.25, 0.5, 0.75, 1.0],
        observed=[1.0, 1.0, 1.0, 1.0, 1.0],
        amplitude_parameter=amplitude_parameter,
    )
    simulation = simulate_amyloid_condition(
        condition=condition,
        parameters=_true_parameters(),
        rtol=1e-9,
        atol=1e-11,
    )

    return AmyloidCondition(
        name=name,
        mtot=mtot,
        timepoints=condition.timepoints,
        observed=simulation["predicted"].to_numpy(),
        amplitude_parameter=amplitude_parameter,
    )


def test_amyloid_aggregation_fit_recovers_shared_rate():
    result = fit_amyloid_aggregation(
        conditions=[
            _synthetic_condition("1000000uM", 1.0, "A1000000"),
            _synthetic_condition("800000uM", 0.8, "A800000"),
        ],
        parameter_specs=_parameter_specs(),
        residual_mode="linear",
        rtol=1e-9,
        atol=1e-11,
    )

    assert result.success
    assert result.fitted_parameters["log_ke"] == pytest.approx(0.0, abs=1e-3)
    assert result.statistics["rss"] == pytest.approx(0.0, abs=1e-10)
    assert set(result.condition_table["condition"]) == {"1000000uM", "800000uM"}


def test_amyloid_aggregation_backend_config_exports_payload(tmp_path):
    condition = _synthetic_condition("1000000uM", 1.0, "A1000000")

    config = {
        "conditions": [
            {
                "name": condition.name,
                "mtot": condition.mtot,
                "timepoints": list(condition.timepoints),
                "observed": list(condition.observed),
                "amplitude_parameter": condition.amplitude_parameter,
            }
        ],
        "parameters": {
            spec.name: {
                "initial_guess": spec.initial_guess,
                "lower_bound": spec.lower_bound,
                "upper_bound": spec.upper_bound,
                "fixed": spec.fixed,
                "fixed_value": spec.fixed_value,
            }
            for spec in _parameter_specs()
            if spec.name != "A800000"
        },
        "residual_mode": "linear",
        "rtol": 1e-9,
        "atol": 1e-11,
        "output_dir": str(tmp_path),
    }

    output = fit_amyloid_aggregation_from_config(config)
    payload = backend_output_payload(
        output,
        workflow="amyloid_aggregation",
        max_rows=3,
    )

    assert output["result"].success
    assert payload["workflow"] == "amyloid_aggregation"
    assert payload["result"]["success"] is True
    assert payload["conditions"][0]["name"] == "1000000uM"
    assert (tmp_path / "fitted_parameters.csv").exists()
    assert (tmp_path / "predicted_curves.csv").exists()

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from odefit.cli import (
    build_initial_condition_specs,
    build_initial_condition_specs_by_model_from_config,
    build_model_specs_from_comparison_config,
    build_parameter_specs,
    build_parameter_specs_by_model_from_config,
    load_fit_config,
    parse_signal_weight_entries,
    read_wide_observable_dataset_with_filtering,
)
from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.multispecies_variable_projection import (
    fit_global_observable_model_multispecies_variable_projection,
)
from odefit.fitting.multispecies_variable_projection_bootstrap import (
    bootstrap_global_observable_multispecies_variable_projection_fit,
)
from odefit.fitting.multispecies_variable_projection_model_comparison import (
    fit_global_observable_multispecies_variable_projection_model_comparison,
)
from odefit.fitting.multispecies_variable_projection_profile_likelihood import (
    fit_multispecies_variable_projection_profile_likelihood,
)
from odefit.fitting.variable_projection import (
    fit_global_observable_model_variable_projection,
)
from odefit.fitting.variable_projection_bootstrap import (
    bootstrap_global_observable_variable_projection_fit,
)
from odefit.fitting.variable_projection_model_comparison import (
    fit_global_observable_variable_projection_model_comparison,
)
from odefit.fitting.variable_projection_profile_likelihood import (
    fit_variable_projection_profile_likelihood,
)
from odefit.model.model_spec import build_model_spec
from odefit.model.ode_generator import generate_ode_lines
from odefit.simulation.solver import simulate_model
from odefit.engines.registry import (
    available_engine_names,
    describe_available_engines,
    get_engine_bundle,
)
from odefit.fitting.amyloid_aggregation import (
    fit_amyloid_aggregation_from_config as _fit_amyloid_aggregation_from_config,
)

def parse_model_text(
    model_text: str,
    *,
    name: str | None = None,
) -> dict:
    model = build_model_spec(
        text=model_text,
        name=name,
    )

    return {
        "name": model.name,
        "species": list(model.species),
        "parameters": list(model.parameters),
        "reactions": [str(reaction) for reaction in model.reactions],
        "ode_lines": generate_ode_lines(model),
    }


def simulate_from_text(
    *,
    model_text: str,
    parameters: dict[str, float],
    initial_conditions: dict[str, float],
    timepoints: list[float],
    name: str | None = None,
) -> pd.DataFrame:
    model = build_model_spec(
        text=model_text,
        name=name,
    )

    result = simulate_model(
        model=model,
        parameters=parameters,
        initial_conditions=initial_conditions,
        timepoints=timepoints,
    )

    dataframe = pd.DataFrame({"time": result.timepoints})

    for species in result.species:
        dataframe[species] = result.get_species_values(species)

    return dataframe


def fit_amyloid_aggregation_from_config(
    config_or_path: dict | str | Path,
):
    """
    Fit the built-in amyloid aggregation workflow from a config dictionary/path.

    This supports the legacy apo aggregation script pattern:
    multiple total-concentration conditions, shared kinetic parameters, optional
    per-condition amplitude parameters, LSODA integration, and log residuals.
    """

    config = _load_config(config_or_path)

    return _fit_amyloid_aggregation_from_config(config)


def _load_config(config_or_path: dict | str | Path) -> dict:
    if isinstance(config_or_path, dict):
        return config_or_path

    return load_fit_config(config_or_path)




def _engine_name_from_config(config: dict) -> str:
    return str(config.get("engine_name", "reference"))
def _read_dataset_from_config(config: dict):
    peak_filtering_settings = {
        "max_missing_fraction": float(config.get("max_missing_fraction", 0.0)),
        "min_initial_intensity": (
            None
            if config.get("min_initial_intensity") is None
            else float(config.get("min_initial_intensity"))
        ),
        "initial_points": int(config.get("initial_points", 1)),
        "min_dynamic_range": (
            None
            if config.get("min_dynamic_range") is None
            else float(config.get("min_dynamic_range"))
        ),
        "interpolate_missing": bool(config.get("interpolate_missing", True)),
    }

    return read_wide_observable_dataset_with_filtering(
        file_path=config["data"],
        time_column=config.get("time_column", "time"),
        signal_columns=config.get("signal_columns"),
        exclude_columns=config.get("exclude_columns"),
        max_missing_fraction=peak_filtering_settings["max_missing_fraction"],
        min_initial_intensity=peak_filtering_settings["min_initial_intensity"],
        initial_points=peak_filtering_settings["initial_points"],
        min_dynamic_range=peak_filtering_settings["min_dynamic_range"],
        interpolate_missing=peak_filtering_settings["interpolate_missing"],
    )


def _settings_from_config(config: dict) -> FitSettings:
    signal_weight_entries = config.get("signal_weight")

    if signal_weight_entries is None:
        signal_weight_entries = config.get("signal_weights")

    return FitSettings(
        species_mapping={},
        use_normalized_data=False,
        method=config.get("method", "trf"),
        loss=config.get("loss", "linear"),
        max_nfev=config.get("max_nfev"),
        rtol=config.get("rtol", 1e-6),
        atol=config.get("atol", 1e-9),
        signal_weights=parse_signal_weight_entries(signal_weight_entries),
    )


def _single_model_from_config(config: dict):
    model_path = config.get("model")
    model_text = config.get("model_text")

    if model_text is not None:
        return build_model_spec(
            text=model_text,
            name=config.get("model_name"),
        )

    if model_path is None:
        raise ValueError("Config requires either 'model' or 'model_text'.")

    return build_model_spec(
        text=Path(model_path).read_text(),
        name=config.get("model_name"),
    )


def _parameter_entries_from_config(config: dict):
    parameter_entries = config.get("parameter")

    if parameter_entries is None:
        parameter_entries = config.get("parameters")

    return parameter_entries


def _initial_entries_from_config(config: dict):
    initial_entries = config.get("initial")

    if initial_entries is None:
        initial_entries = config.get("initial_conditions")

    return initial_entries


def fit_global_observables_from_config(
    config_or_path: dict | str | Path,
):
    config = _load_config(config_or_path)
    engine_name = _engine_name_from_config(config)
    model = _single_model_from_config(config)
    dataset, filtering_result = _read_dataset_from_config(config)

    parameter_specs = build_parameter_specs(
        model=model,
        parameter_entries=_parameter_entries_from_config(config),
        default_guess=float(config.get("default_parameter_guess", 0.1)),
        default_lower=float(config.get("default_parameter_lower", 0.0)),
        default_upper=float(config.get("default_parameter_upper", 100.0)),
    )

    initial_condition_specs = build_initial_condition_specs(
        model=model,
        initial_entries=_initial_entries_from_config(config),
    )

    settings = _settings_from_config(config)

    if config.get("use_multispecies_variable_projection", False):
        result = fit_global_observable_model_multispecies_variable_projection(
            model=model,
            dataset=dataset,
            parameter_specs=parameter_specs,
            initial_condition_specs=initial_condition_specs,
            observed_species=config["observed_species"],
            settings=settings,
            signal_columns=dataset.signal_columns,
            fit_offset=bool(config.get("fit_offset", True)),
            backend=str(config.get("variable_projection_backend", "numpy")),
            method=str(config.get("variable_projection_method", "LSODA")),
            engine_name=engine_name,
        )
    else:
        result = fit_global_observable_model_variable_projection(
            model=model,
            dataset=dataset,
            parameter_specs=parameter_specs,
            initial_condition_specs=initial_condition_specs,
            observed_species=config.get("observed_species", "A"),
            settings=settings,
            signal_columns=dataset.signal_columns,
            fit_scale=bool(config.get("fit_scale", True)),
            fit_offset=bool(config.get("fit_offset", True)),
            backend=str(config.get("variable_projection_backend", "numpy")),
            method=str(config.get("variable_projection_method", "LSODA")),
            engine_name=engine_name,
        )

    return {
        "result": result,
        "dataset": dataset,
        "filtering_result": filtering_result,
    }


def compare_global_observables_from_config(
    config_or_path: dict | str | Path,
):
    config = _load_config(config_or_path)
    engine_name = _engine_name_from_config(config)
    dataset, filtering_result = _read_dataset_from_config(config)
    models = build_model_specs_from_comparison_config(config)
    settings = _settings_from_config(config)

    parameter_specs_by_model = build_parameter_specs_by_model_from_config(
        models=models,
        config=config,
        default_guess=float(config.get("default_parameter_guess", 0.1)),
        default_lower=float(config.get("default_parameter_lower", 0.0)),
        default_upper=float(config.get("default_parameter_upper", 100.0)),
    )

    initial_condition_specs_by_model = (
        build_initial_condition_specs_by_model_from_config(
            models=models,
            config=config,
        )
    )

    if config.get("use_multispecies_variable_projection", False):
        result = (
            fit_global_observable_multispecies_variable_projection_model_comparison(
                models=models,
                dataset=dataset,
                parameter_specs_by_model=parameter_specs_by_model,
                initial_condition_specs_by_model=initial_condition_specs_by_model,
                observed_species_by_model=config.get(
                    "observed_species_by_model",
                    config.get("observed_species"),
                ),
                settings=settings,
                signal_columns=dataset.signal_columns,
                fit_offset=bool(config.get("fit_offset", True)),
                backend=str(config.get("variable_projection_backend", "numpy")),
                method=str(config.get("variable_projection_method", "LSODA")),
                sort_by=config.get("sort_by", "bic"),
                engine_name=engine_name,
            )
        )
    else:
        result = fit_global_observable_variable_projection_model_comparison(
            models=models,
            dataset=dataset,
            parameter_specs_by_model=parameter_specs_by_model,
            initial_condition_specs_by_model=initial_condition_specs_by_model,
            observed_species_by_model=config.get(
                "observed_species_by_model",
                config.get("observed_species", "A"),
            ),
            signal_columns=dataset.signal_columns,
            fit_scale=bool(config.get("fit_scale", True)),
            fit_offset=bool(config.get("fit_offset", True)),
            backend=str(config.get("variable_projection_backend", "numpy")),
            method=str(config.get("variable_projection_method", "LSODA")),
            sort_by=config.get("sort_by", "bic"),
            engine_name=engine_name,
        )

    return {
        "result": result,
        "dataset": dataset,
        "filtering_result": filtering_result,
        "models": models,
    }


def bootstrap_global_observables_from_config(
    config_or_path: dict | str | Path,
):
    config = _load_config(config_or_path)
    engine_name = _engine_name_from_config(config)
    model = _single_model_from_config(config)
    dataset, filtering_result = _read_dataset_from_config(config)
    settings = _settings_from_config(config)

    parameter_specs = build_parameter_specs(
        model=model,
        parameter_entries=_parameter_entries_from_config(config),
        default_guess=float(config.get("default_parameter_guess", 0.1)),
        default_lower=float(config.get("default_parameter_lower", 0.0)),
        default_upper=float(config.get("default_parameter_upper", 100.0)),
    )

    initial_condition_specs = build_initial_condition_specs(
        model=model,
        initial_entries=_initial_entries_from_config(config),
    )

    if config.get("use_multispecies_variable_projection", False):
        result = bootstrap_global_observable_multispecies_variable_projection_fit(
            model=model,
            dataset=dataset,
            parameter_specs=parameter_specs,
            initial_condition_specs=initial_condition_specs,
            observed_species=config["observed_species"],
            settings=settings,
            signal_columns=dataset.signal_columns,
            fit_offset=bool(config.get("fit_offset", True)),
            backend=str(config.get("variable_projection_backend", "numpy")),
            method=str(config.get("variable_projection_method", "LSODA")),
            n_bootstrap=int(config.get("n_bootstrap", 100)),
            n_workers=int(config.get("n_workers", 1)),
            random_seed=config.get("random_seed"),
            confidence_level=float(config.get("confidence_level", 0.95)),
            show_progress=bool(config.get("show_progress", True)),
            engine_name=engine_name,
        )
    else:
        result = bootstrap_global_observable_variable_projection_fit(
            model=model,
            dataset=dataset,
            parameter_specs=parameter_specs,
            initial_condition_specs=initial_condition_specs,
            observed_species=config.get("observed_species", "A"),
            settings=settings,
            signal_columns=dataset.signal_columns,
            fit_scale=bool(config.get("fit_scale", True)),
            fit_offset=bool(config.get("fit_offset", True)),
            backend=str(config.get("variable_projection_backend", "numpy")),
            method=str(config.get("variable_projection_method", "LSODA")),
            n_bootstrap=int(config.get("n_bootstrap", 100)),
            n_workers=int(config.get("n_workers", 1)),
            random_seed=config.get("random_seed"),
            confidence_level=float(config.get("confidence_level", 0.95)),
            show_progress=bool(config.get("show_progress", True)),
            engine_name=engine_name,
        )

    return {
        "result": result,
        "dataset": dataset,
        "filtering_result": filtering_result,
    }

def get_backend_engine_capabilities() -> dict:
    """
    Return a GUI-friendly description of available backend engines.

    This function is intentionally JSON-serializable and stable enough for GUI
    contract tests. Optional engines that cannot be constructed are reported
    with available=False rather than raising.
    """

    raw_descriptions = describe_available_engines()

    engines = []

    for entry in raw_descriptions:
        name = str(entry.get("name", "unknown"))
        available = bool(entry.get("available", False))

        raw_capabilities = entry.get("capabilities", {})
        normalized_capabilities = {}

        if isinstance(raw_capabilities, dict):
            for component_name, capability in raw_capabilities.items():
                if isinstance(capability, dict):
                    normalized_capabilities[component_name] = {
                        "name": capability.get("name", component_name),
                        "single_species_projection": bool(
                            capability.get("single_species_projection", False)
                        ),
                        "multispecies_projection": bool(
                            capability.get("multispecies_projection", False)
                        ),
                        "supports_parallel": bool(
                            capability.get("supports_parallel", False)
                        ),
                        "supports_jit": bool(
                            capability.get("supports_jit", False)
                        ),
                        "supports_gpu": bool(
                            capability.get("supports_gpu", False)
                        ),
                        "supports_autodiff": bool(
                            capability.get("supports_autodiff", False)
                        ),
                        "notes": str(capability.get("notes", "")),
                    }
                else:
                    normalized_capabilities[component_name] = capability

        engine_payload = {
            "name": name,
            "available": available,
            "is_default": name == "reference",
            "capabilities": normalized_capabilities,
            "error_type": entry.get("error_type"),
            "error_message": entry.get("error_message"),
        }

        engines.append(engine_payload)

    return {
        "default_engine": "reference",
        "available_engine_names": available_engine_names(),
        "engines": engines,
    }


def validate_backend_engine_name(engine_name: str) -> dict:
    """
    Validate one backend engine name and return a GUI-friendly payload.

    This is useful for forms/config validation before launching a fit.
    """

    try:
        bundle = get_engine_bundle(engine_name)
    except Exception as exc:
        return {
            "engine_name": engine_name,
            "valid": False,
            "available": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

    capabilities = {
        component_name: capability.__dict__
        for component_name, capability in bundle.capabilities().items()
    }

    return {
        "engine_name": engine_name,
        "valid": True,
        "available": True,
        "resolved_name": bundle.name,
        "capabilities": capabilities,
    }

def profile_likelihood_global_observables_from_config(
    config_or_path: dict | str | Path,
):
    config = _load_config(config_or_path)
    engine_name = _engine_name_from_config(config)
    model = _single_model_from_config(config)
    dataset, filtering_result = _read_dataset_from_config(config)
    settings = _settings_from_config(config)

    parameter_specs = build_parameter_specs(
        model=model,
        parameter_entries=_parameter_entries_from_config(config),
        default_guess=float(config.get("default_parameter_guess", 0.1)),
        default_lower=float(config.get("default_parameter_lower", 0.0)),
        default_upper=float(config.get("default_parameter_upper", 100.0)),
    )

    initial_condition_specs = build_initial_condition_specs(
        model=model,
        initial_entries=_initial_entries_from_config(config),
    )

    if config.get("use_multispecies_variable_projection", False):
        result = fit_multispecies_variable_projection_profile_likelihood(
            model=model,
            dataset=dataset,
            parameter_specs=parameter_specs,
            initial_condition_specs=initial_condition_specs,
            observed_species=config["observed_species"],
            settings=settings,
            signal_columns=dataset.signal_columns,
            fit_offset=bool(config.get("fit_offset", True)),
            backend=str(config.get("variable_projection_backend", "numpy")),
            method=str(config.get("variable_projection_method", "LSODA")),
            profile_parameters=config.get("profile_parameters"),
            n_points=int(config.get("profile_n_points", 15)),
            span_factor=float(config.get("profile_span_factor", 10.0)),
            log_space=bool(config.get("profile_log_space", True)),
            show_progress=bool(config.get("show_progress", True)),
            engine_name=engine_name,
        )
    else:
        result = fit_variable_projection_profile_likelihood(
            model=model,
            dataset=dataset,
            parameter_specs=parameter_specs,
            initial_condition_specs=initial_condition_specs,
            observed_species=config.get("observed_species", "A"),
            settings=settings,
            signal_columns=dataset.signal_columns,
            fit_scale=bool(config.get("fit_scale", True)),
            fit_offset=bool(config.get("fit_offset", True)),
            backend=str(config.get("variable_projection_backend", "numpy")),
            method=str(config.get("variable_projection_method", "LSODA")),
            profile_parameters=config.get("profile_parameters"),
            n_points=int(config.get("profile_n_points", 15)),
            span_factor=float(config.get("profile_span_factor", 10.0)),
            log_space=bool(config.get("profile_log_space", True)),
            show_progress=bool(config.get("show_progress", True)),
            engine_name=engine_name,
        )

    return {
        "result": result,
        "dataset": dataset,
        "filtering_result": filtering_result,
    }

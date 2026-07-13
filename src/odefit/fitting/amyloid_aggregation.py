from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from odefit.export.json_export import to_jsonable
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.fitting.parameter_vector import (
    build_bounds,
    build_initial_vector,
    get_free_parameter_specs,
    validate_parameter_specs,
    vector_to_parameter_dict,
)
from odefit.fitting.statistics import calculate_fit_statistics


@dataclass
class AmyloidCondition:
    """
    One experimental total-concentration condition for amyloid aggregation fits.
    """

    name: str
    mtot: float
    timepoints: np.ndarray
    observed: np.ndarray
    fit: bool = True
    amplitude_parameter: str | None = None
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AmyloidAggregationFitResult:
    """
    Fit result for the amyloid aggregation multi-condition workflow.
    """

    success: bool
    message: str
    fitted_parameters: dict[str, float]
    initial_parameters: dict[str, float]
    statistics: dict[str, Any]
    residuals: np.ndarray
    residual_vector: np.ndarray
    condition_table: pd.DataFrame
    observable_table: pd.DataFrame
    simulation_dataframe: pd.DataFrame
    predicted_dataframe: pd.DataFrame
    residuals_dataframe: pd.DataFrame
    cost: float
    nfev: int
    status: int | None = None
    optimality: float | None = None
    active_mask: np.ndarray | None = None
    njev: int | None = None
    fitted_initial_conditions: dict[str, float] | None = None
    initial_conditions: dict[str, float] | None = None


def smooth_max(
    value,
    *,
    eps: float = 1e-12,
):
    """
    Smooth approximation to max(value, 0).
    """

    return 0.5 * (value + np.sqrt(value * value + eps))


def _rate_parameter(
    parameters: dict[str, float],
    *,
    log_name: str,
    direct_name: str,
) -> float:
    if log_name in parameters:
        return float(10 ** parameters[log_name])

    if direct_name in parameters:
        return float(parameters[direct_name])

    raise ValueError(
        f"Missing rate parameter. Provide either {log_name!r} or {direct_name!r}."
    )


def amyloid_aggregation_rhs(
    t: float,
    y: np.ndarray,
    *,
    mtot: float,
    parameters: dict[str, float],
    smooth_eps: float = 1e-12,
) -> list[float]:
    """
    Amyloid aggregation RHS matching the pasted legacy script.

    States:
        P:
            number of fibril ends
        Fm:
            fibril mass in monomer units
    """

    del t

    fibril_mass = float(y[1])

    kn = _rate_parameter(parameters, log_name="log_kn", direct_name="kn")
    k2 = _rate_parameter(parameters, log_name="log_k2", direct_name="k2")
    ke = _rate_parameter(parameters, log_name="log_ke", direct_name="ke")

    nc = float(parameters["nc"])
    n2 = float(parameters.get("n2", 1.0))
    cm = float(parameters["cm"])

    effective_monomer = smooth_max(
        mtot - fibril_mass - cm,
        eps=smooth_eps,
    )

    d_ends = kn * effective_monomer**nc + (
        k2 * effective_monomer**n2
    ) * fibril_mass
    d_fibril_mass = (2.0 * ke * effective_monomer) * d_ends

    return [d_ends, d_fibril_mass]


def simulate_amyloid_condition(
    condition: AmyloidCondition,
    parameters: dict[str, float],
    *,
    method: str = "LSODA",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    initial_ends: float = 0.0,
    initial_fibril_mass: float = 0.0,
    smooth_eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Simulate one amyloid aggregation condition.
    """

    timepoints = np.asarray(condition.timepoints, dtype=float)

    if len(timepoints) < 2:
        raise ValueError("Amyloid simulation requires at least two timepoints.")

    if np.any(np.diff(timepoints) <= 0):
        raise ValueError(
            f"Timepoints must be strictly increasing for condition {condition.name}."
        )

    solution = solve_ivp(
        fun=lambda t, y: amyloid_aggregation_rhs(
            t,
            y,
            mtot=condition.mtot,
            parameters=parameters,
            smooth_eps=smooth_eps,
        ),
        t_span=(float(timepoints[0]), float(timepoints[-1])),
        y0=[float(initial_ends), float(initial_fibril_mass)],
        t_eval=timepoints,
        method=method,
        rtol=rtol,
        atol=atol,
    )

    if not solution.success:
        raise RuntimeError(
            f"ODE solve failed for condition {condition.name}: {solution.message}"
        )

    ends = solution.y[0]
    fibril_mass = solution.y[1]
    monomer_fraction = (condition.mtot - fibril_mass) / condition.mtot

    amplitude = 1.0

    if condition.amplitude_parameter is not None:
        amplitude = float(parameters.get(condition.amplitude_parameter, 1.0))

    predicted_signal = amplitude * monomer_fraction

    return pd.DataFrame(
        {
            "condition": condition.name,
            "time": timepoints,
            "mtot": float(condition.mtot),
            "P": ends,
            "Fm": fibril_mass,
            "monomer_fraction": monomer_fraction,
            "predicted": predicted_signal,
            "amplitude": amplitude,
        }
    )


def _condition_residuals(
    *,
    observed: np.ndarray,
    predicted: np.ndarray,
    residual_mode: str,
    log_epsilon: float,
    weight: float,
) -> np.ndarray:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if residual_mode == "linear":
        residuals = observed - predicted
    elif residual_mode == "relative":
        residuals = (observed - predicted) / np.maximum(np.abs(observed), log_epsilon)
    elif residual_mode == "log":
        residuals = np.log(np.maximum(observed, log_epsilon)) - np.log(
            np.maximum(predicted, log_epsilon)
        )
    else:
        raise ValueError(
            "residual_mode must be one of: linear, relative, log. "
            f"Got: {residual_mode}"
        )

    return float(weight) * residuals


def _objective_vector(
    vector: np.ndarray,
    *,
    parameter_specs: list[ParameterSpec],
    conditions: list[AmyloidCondition],
    residual_mode: str,
    log_epsilon: float,
    solver_method: str,
    rtol: float,
    atol: float,
    initial_ends: float,
    initial_fibril_mass: float,
    smooth_eps: float,
    failure_penalty: float,
) -> np.ndarray:
    parameters = vector_to_parameter_dict(vector, parameter_specs)
    residual_blocks = []

    for condition in conditions:
        if not condition.fit:
            continue

        try:
            simulation = simulate_amyloid_condition(
                condition=condition,
                parameters=parameters,
                method=solver_method,
                rtol=rtol,
                atol=atol,
                initial_ends=initial_ends,
                initial_fibril_mass=initial_fibril_mass,
                smooth_eps=smooth_eps,
            )
            residual_blocks.append(
                _condition_residuals(
                    observed=condition.observed,
                    predicted=simulation["predicted"].to_numpy(),
                    residual_mode=residual_mode,
                    log_epsilon=log_epsilon,
                    weight=condition.weight,
                )
            )
        except Exception:
            residual_blocks.append(
                np.full(len(condition.observed), float(failure_penalty), dtype=float)
            )

    if not residual_blocks:
        raise ValueError("At least one amyloid condition must have fit=True.")

    return np.concatenate(residual_blocks)


def _build_output_tables(
    *,
    conditions: list[AmyloidCondition],
    parameters: dict[str, float],
    residual_mode: str,
    log_epsilon: float,
    solver_method: str,
    rtol: float,
    atol: float,
    initial_ends: float,
    initial_fibril_mass: float,
    smooth_eps: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    simulation_tables = []
    prediction_tables = []
    residual_tables = []
    condition_rows = []
    fit_residual_blocks = []

    for condition in conditions:
        simulation = simulate_amyloid_condition(
            condition=condition,
            parameters=parameters,
            method=solver_method,
            rtol=rtol,
            atol=atol,
            initial_ends=initial_ends,
            initial_fibril_mass=initial_fibril_mass,
            smooth_eps=smooth_eps,
        )

        residuals = _condition_residuals(
            observed=condition.observed,
            predicted=simulation["predicted"].to_numpy(),
            residual_mode=residual_mode,
            log_epsilon=log_epsilon,
            weight=condition.weight,
        )

        if condition.fit:
            fit_residual_blocks.append(residuals)

        prediction_table = pd.DataFrame(
            {
                "condition": condition.name,
                "time": condition.timepoints,
                "mtot": float(condition.mtot),
                "observed": condition.observed,
                "predicted": simulation["predicted"].to_numpy(),
                "used_in_fit": bool(condition.fit),
            }
        )

        residual_table = pd.DataFrame(
            {
                "condition": condition.name,
                "time": condition.timepoints,
                "mtot": float(condition.mtot),
                "residual": residuals,
                "residual_mode": residual_mode,
                "used_in_fit": bool(condition.fit),
            }
        )

        amplitude_value = 1.0

        if condition.amplitude_parameter is not None:
            amplitude_value = float(parameters.get(condition.amplitude_parameter, 1.0))

        condition_rows.append(
            {
                "condition": condition.name,
                "mtot": float(condition.mtot),
                "n_timepoints": int(len(condition.timepoints)),
                "used_in_fit": bool(condition.fit),
                "weight": float(condition.weight),
                "amplitude_parameter": condition.amplitude_parameter,
                "amplitude_value": amplitude_value,
            }
        )

        simulation_tables.append(simulation)
        prediction_tables.append(prediction_table)
        residual_tables.append(residual_table)

    residual_vector = np.concatenate(fit_residual_blocks)

    return (
        pd.DataFrame(condition_rows),
        pd.concat(simulation_tables, ignore_index=True),
        pd.concat(prediction_tables, ignore_index=True),
        pd.concat(residual_tables, ignore_index=True),
        residual_vector,
    )


def _initial_parameter_dict(
    parameter_specs: list[ParameterSpec],
) -> dict[str, float]:
    values = {}

    for parameter in parameter_specs:
        if parameter.fixed:
            values[parameter.name] = float(parameter.fixed_value)
        else:
            values[parameter.name] = float(parameter.initial_guess)

    return values


def fit_amyloid_aggregation(
    *,
    conditions: list[AmyloidCondition],
    parameter_specs: list[ParameterSpec],
    residual_mode: str = "log",
    log_epsilon: float = 1e-12,
    method: str = "trf",
    loss: str = "linear",
    max_nfev: int | None = None,
    solver_method: str = "LSODA",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    initial_ends: float = 0.0,
    initial_fibril_mass: float = 0.0,
    smooth_eps: float = 1e-12,
    failure_penalty: float = 1e12,
) -> AmyloidAggregationFitResult:
    """
    Fit the amyloid aggregation model across multiple total concentrations.
    """

    validate_parameter_specs(parameter_specs)

    if not get_free_parameter_specs(parameter_specs):
        raise ValueError("At least one amyloid parameter must be free to fit.")

    if not any(condition.fit for condition in conditions):
        raise ValueError("At least one amyloid condition must have fit=True.")

    initial_vector = build_initial_vector(parameter_specs)
    bounds = build_bounds(parameter_specs)

    optimizer_result = least_squares(
        fun=lambda vector: _objective_vector(
            vector,
            parameter_specs=parameter_specs,
            conditions=conditions,
            residual_mode=residual_mode,
            log_epsilon=log_epsilon,
            solver_method=solver_method,
            rtol=rtol,
            atol=atol,
            initial_ends=initial_ends,
            initial_fibril_mass=initial_fibril_mass,
            smooth_eps=smooth_eps,
            failure_penalty=failure_penalty,
        ),
        x0=initial_vector,
        bounds=bounds,
        method=method,
        loss=loss,
        max_nfev=max_nfev,
    )

    fitted_parameters = vector_to_parameter_dict(optimizer_result.x, parameter_specs)
    initial_parameters = _initial_parameter_dict(parameter_specs)

    (
        condition_table,
        simulation_dataframe,
        predicted_dataframe,
        residuals_dataframe,
        residual_vector,
    ) = _build_output_tables(
        conditions=conditions,
        parameters=fitted_parameters,
        residual_mode=residual_mode,
        log_epsilon=log_epsilon,
        solver_method=solver_method,
        rtol=rtol,
        atol=atol,
        initial_ends=initial_ends,
        initial_fibril_mass=initial_fibril_mass,
        smooth_eps=smooth_eps,
    )

    statistics = calculate_fit_statistics(
        residuals=residual_vector,
        number_of_parameters=len(get_free_parameter_specs(parameter_specs)),
    )
    statistics.update(
        {
            "n_conditions": float(len(conditions)),
            "n_fit_conditions": float(sum(condition.fit for condition in conditions)),
            "residual_mode": residual_mode,
        }
    )

    return AmyloidAggregationFitResult(
        success=bool(optimizer_result.success),
        message=str(optimizer_result.message),
        fitted_parameters=fitted_parameters,
        initial_parameters=initial_parameters,
        statistics=statistics,
        residuals=residual_vector,
        residual_vector=residual_vector,
        condition_table=condition_table,
        observable_table=condition_table,
        simulation_dataframe=simulation_dataframe,
        predicted_dataframe=predicted_dataframe,
        residuals_dataframe=residuals_dataframe,
        cost=float(optimizer_result.cost),
        nfev=int(optimizer_result.nfev),
        status=int(optimizer_result.status),
        optimality=float(optimizer_result.optimality),
        active_mask=optimizer_result.active_mask,
        njev=optimizer_result.njev,
        fitted_initial_conditions={
            "P": float(initial_ends),
            "Fm": float(initial_fibril_mass),
        },
        initial_conditions={
            "P": float(initial_ends),
            "Fm": float(initial_fibril_mass),
        },
    )


def _column_values(
    dataframe: pd.DataFrame,
    selector,
) -> np.ndarray:
    if isinstance(selector, int):
        return dataframe.iloc[:, selector].to_numpy(dtype=float)

    if isinstance(selector, str) and selector.isdigit():
        return dataframe.iloc[:, int(selector)].to_numpy(dtype=float)

    if selector in dataframe.columns:
        return dataframe[selector].to_numpy(dtype=float)

    raise ValueError(f"Column not found in condition data: {selector}")


def _read_condition_file(
    entry: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    path = Path(entry["data"])

    if not path.exists():
        raise FileNotFoundError(f"Amyloid condition data file does not exist: {path}")

    has_header = entry.get("has_header")

    if has_header is None:
        has_header = path.suffix.lower() == ".csv"

    delimiter = entry.get("delimiter")

    if delimiter == "whitespace":
        separator = r"\s+"
    elif delimiter == "tab":
        separator = "\t"
    elif delimiter is None and path.suffix.lower() != ".csv":
        separator = r"\s+"
    elif delimiter in {None, "csv"}:
        separator = ","
    else:
        separator = str(delimiter)

    dataframe = pd.read_csv(
        path,
        sep=separator,
        header=0 if has_header else None,
        engine="python",
    )

    time_column = entry.get("time_column", "time" if has_header else 1)
    signal_column = entry.get("signal_column", "signal" if has_header else 0)

    timepoints = _column_values(dataframe, time_column)
    observed = _column_values(dataframe, signal_column)

    return timepoints, observed


def _mtot_label(mtot: float) -> str:
    micromolar = float(mtot) * 1e6

    if abs(micromolar - round(micromolar)) < 1e-9:
        return str(int(round(micromolar)))

    return f"{micromolar:g}".replace(".", "p")


def build_amyloid_conditions_from_config(
    config: dict[str, Any],
) -> list[AmyloidCondition]:
    """
    Build amyloid fit conditions from config dictionaries.
    """

    condition_entries = config.get("conditions")

    if not condition_entries:
        raise ValueError("Amyloid aggregation config requires 'conditions'.")

    conditions = []

    for index, entry in enumerate(condition_entries):
        mtot = float(entry["mtot"])
        name = str(entry.get("name", f"{_mtot_label(mtot)}uM"))

        if "timepoints" in entry and "observed" in entry:
            timepoints = np.asarray(entry["timepoints"], dtype=float)
            observed = np.asarray(entry["observed"], dtype=float)
        else:
            timepoints, observed = _read_condition_file(entry)

        time_scale = float(entry.get("time_scale", 1.0))
        signal_scale = float(entry.get("signal_scale", 1.0))

        timepoints = time_scale * np.asarray(timepoints, dtype=float)
        observed = signal_scale * np.asarray(observed, dtype=float)

        time_min = entry.get("time_min")
        time_max = entry.get("time_max", config.get("max_time"))

        mask = np.ones(len(timepoints), dtype=bool)

        if time_min is not None:
            mask &= timepoints >= float(time_min)

        if time_max is not None:
            mask &= timepoints <= float(time_max)

        timepoints = timepoints[mask]
        observed = observed[mask]

        order = np.argsort(timepoints)
        timepoints = timepoints[order]
        observed = observed[order]

        if len(timepoints) != len(observed):
            raise ValueError(f"Condition {name} time and signal lengths differ.")

        if len(timepoints) < 2:
            raise ValueError(f"Condition {name} has fewer than two timepoints.")

        if np.any(np.diff(timepoints) <= 0):
            raise ValueError(f"Condition {name} contains duplicate timepoints.")

        conditions.append(
            AmyloidCondition(
                name=name,
                mtot=mtot,
                timepoints=timepoints,
                observed=observed,
                fit=bool(entry.get("fit", True)),
                amplitude_parameter=entry.get(
                    "amplitude_parameter",
                    f"A{_mtot_label(mtot)}",
                ),
                weight=float(entry.get("weight", 1.0)),
                metadata={
                    "index": index,
                    "data": entry.get("data"),
                    "time_column": entry.get("time_column"),
                    "signal_column": entry.get("signal_column"),
                },
            )
        )

    return conditions


def _parameter_spec_from_config(
    name: str,
    values: dict[str, Any],
    *,
    base: ParameterSpec | None = None,
) -> ParameterSpec:
    mode = str(values.get("mode", "")).lower()
    fixed = bool(values.get("fixed", base.fixed if base else False) or mode == "fixed")
    fixed_value = values.get("fixed_value", values.get("value"))

    if fixed and fixed_value is None:
        fixed_value = values.get(
            "initial_guess",
            base.fixed_value if base else None,
        )

    initial_guess = values.get(
        "initial_guess",
        fixed_value
        if fixed_value is not None
        else (base.initial_guess if base else None),
    )

    if initial_guess is None:
        raise ValueError(f"Parameter {name} requires initial_guess or value.")

    return ParameterSpec(
        name=name,
        initial_guess=float(initial_guess),
        lower_bound=float(
            values.get(
                "lower_bound",
                values.get("min", base.lower_bound if base else 0.0),
            )
        ),
        upper_bound=float(
            values.get(
                "upper_bound",
                values.get("max", base.upper_bound if base else np.inf),
            )
        ),
        fixed=fixed,
        fixed_value=None if fixed_value is None else float(fixed_value),
        tied_to=values.get("tied_to", base.tied_to if base else None),
    )


def default_amyloid_parameter_specs(
    conditions: list[AmyloidCondition],
) -> list[ParameterSpec]:
    """
    Defaults matching the pasted script.
    """

    specs = [
        ParameterSpec("log_kn", initial_guess=4.0, lower_bound=-12.0, upper_bound=12.0),
        ParameterSpec("log_k2", initial_guess=2.0, lower_bound=-12.0, upper_bound=8.0),
        ParameterSpec("log_ke", initial_guess=4.0, lower_bound=-12.0, upper_bound=8.0),
        ParameterSpec("nc", initial_guess=2.0, lower_bound=1.0, upper_bound=5.0),
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
            initial_guess=35e-6,
            lower_bound=5e-6,
            upper_bound=50e-6,
        ),
    ]

    existing_names = {spec.name for spec in specs}

    for condition in conditions:
        if condition.amplitude_parameter is None:
            continue

        if condition.amplitude_parameter in existing_names:
            continue

        specs.append(
            ParameterSpec(
                condition.amplitude_parameter,
                initial_guess=1.0,
                lower_bound=0.5,
                upper_bound=1.5,
                fixed=True,
                fixed_value=1.0,
            )
        )
        existing_names.add(condition.amplitude_parameter)

    return specs


def build_amyloid_parameter_specs_from_config(
    config: dict[str, Any],
    conditions: list[AmyloidCondition],
) -> list[ParameterSpec]:
    specs_by_name = {
        spec.name: spec
        for spec in default_amyloid_parameter_specs(conditions)
    }

    for name, values in config.get("parameters", {}).items():
        parameter_name = str(name)
        specs_by_name[parameter_name] = _parameter_spec_from_config(
            parameter_name,
            values,
            base=specs_by_name.get(parameter_name),
        )

    for name, values in config.get("amplitudes", {}).items():
        parameter_name = str(name)
        specs_by_name[parameter_name] = _parameter_spec_from_config(
            parameter_name,
            values,
            base=specs_by_name.get(parameter_name),
        )

    ordered_names = list(specs_by_name)
    return [specs_by_name[name] for name in ordered_names]


def fit_amyloid_aggregation_from_config(
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Fit amyloid aggregation from a GUI/CLI-friendly config dictionary.
    """

    conditions = build_amyloid_conditions_from_config(config)
    parameter_specs = build_amyloid_parameter_specs_from_config(config, conditions)

    result = fit_amyloid_aggregation(
        conditions=conditions,
        parameter_specs=parameter_specs,
        residual_mode=str(config.get("residual_mode", "log")),
        log_epsilon=float(config.get("log_epsilon", 1e-12)),
        method=str(config.get("method", "trf")),
        loss=str(config.get("loss", "linear")),
        max_nfev=(
            None
            if config.get("max_nfev") is None
            else int(config["max_nfev"])
        ),
        solver_method=str(
            config.get("solver_method", config.get("ode_method", "LSODA"))
        ),
        rtol=float(config.get("rtol", 1e-6)),
        atol=float(config.get("atol", 1e-9)),
        initial_ends=float(config.get("initial_ends", 0.0)),
        initial_fibril_mass=float(config.get("initial_fibril_mass", 0.0)),
        smooth_eps=float(config.get("smooth_eps", 1e-12)),
        failure_penalty=float(config.get("failure_penalty", 1e12)),
    )

    output = {
        "result": result,
        "conditions": [
            {
                "name": condition.name,
                "mtot": float(condition.mtot),
                "n_timepoints": int(len(condition.timepoints)),
                "fit": bool(condition.fit),
                "amplitude_parameter": condition.amplitude_parameter,
                "weight": float(condition.weight),
                "metadata": dict(condition.metadata),
            }
            for condition in conditions
        ],
        "parameter_specs": parameter_specs,
    }

    output_dir = config.get("output_dir")

    if output_dir:
        output["written_files"] = export_amyloid_aggregation_fit(
            result=result,
            output_dir=output_dir,
        )

    return output


def export_amyloid_aggregation_fit(
    *,
    result: AmyloidAggregationFitResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Export amyloid aggregation fit tables.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: dict[str, Path] = {}

    parameter_table = pd.DataFrame(
        [
            {
                "parameter": name,
                "initial_value": result.initial_parameters.get(name),
                "fitted_value": value,
            }
            for name, value in result.fitted_parameters.items()
        ]
    )

    written_files["fitted_parameters"] = output_path / "fitted_parameters.csv"
    parameter_table.to_csv(written_files["fitted_parameters"], index=False)

    written_files["fit_statistics"] = output_path / "fit_statistics.csv"
    pd.DataFrame([result.statistics]).to_csv(
        written_files["fit_statistics"],
        index=False,
    )

    written_files["conditions"] = output_path / "conditions.csv"
    result.condition_table.to_csv(written_files["conditions"], index=False)

    written_files["simulated_curves"] = output_path / "simulated_curves.csv"
    result.simulation_dataframe.to_csv(written_files["simulated_curves"], index=False)

    written_files["predicted_curves"] = output_path / "predicted_curves.csv"
    result.predicted_dataframe.to_csv(written_files["predicted_curves"], index=False)

    written_files["residuals"] = output_path / "residuals.csv"
    result.residuals_dataframe.to_csv(written_files["residuals"], index=False)

    summary = {
        "success": result.success,
        "message": result.message,
        "fitted_parameters": result.fitted_parameters,
        "statistics": result.statistics,
        "nfev": result.nfev,
        "cost": result.cost,
        "status": result.status,
        "optimality": result.optimality,
    }

    written_files["summary"] = output_path / "summary.json"
    written_files["summary"].write_text(
        json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n"
    )

    return written_files

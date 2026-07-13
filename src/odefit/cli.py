from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from odefit.data.dataset import Dataset
from odefit.data.peak_filtering import build_peak_filtering_table
from odefit.export.bundle_export import export_fit_bundle
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.amyloid_aggregation import (
    fit_amyloid_aggregation_from_config,
)
from odefit.fitting.global_observable_model_comparison import (
    build_model_specs_from_texts,
    export_global_observable_model_comparison,
    fit_global_observable_model_comparison,
)
from odefit.fitting.global_observable_multistart_model_comparison import (
    export_global_observable_multistart_model_comparison,
    fit_global_observable_multistart_model_comparison,
)
from odefit.fitting.global_observables import (
    build_shared_species_observable_specs,
    fit_global_observable_model,
    read_wide_observable_dataset_with_filtering,
)
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.multispecies_variable_projection import (
    export_multispecies_variable_projection_fit,
    fit_global_observable_model_multispecies_variable_projection,
)
from odefit.fitting.multispecies_variable_projection_bootstrap import (
    bootstrap_global_observable_multispecies_variable_projection_fit,
    export_multispecies_variable_projection_bootstrap_result,
)
from odefit.fitting.multispecies_variable_projection_model_comparison import (
    export_multispecies_variable_projection_model_comparison,
    fit_global_observable_multispecies_variable_projection_model_comparison,
)
from odefit.fitting.multispecies_variable_projection_multistart import (
    export_multispecies_variable_projection_multistart_summary,
    fit_global_observable_model_multispecies_variable_projection_multistart,
)
from odefit.fitting.multispecies_variable_projection_multistart_model_comparison import (
    export_multispecies_variable_projection_multistart_model_comparison,
    fit_global_observable_multispecies_variable_projection_multistart_model_comparison,
)
from odefit.fitting.multispecies_variable_projection_profile_likelihood import (
    export_multispecies_profile_likelihood_result,
    fit_multispecies_variable_projection_profile_likelihood,
)
from odefit.fitting.multistart import (
    export_multistart_comparison,
    fit_multistart,
)
from odefit.fitting.optimizer import fit_model
from odefit.fitting.parallel_multistart import (
    export_parallel_multistart_summary,
    fit_multistart_parallel,
)
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.fitting.variable_projection import (
    export_variable_projection_fit,
    fit_global_observable_model_variable_projection,
)
from odefit.fitting.variable_projection_bootstrap import (
    bootstrap_global_observable_variable_projection_fit,
    export_variable_projection_bootstrap_result,
)
from odefit.fitting.variable_projection_model_comparison import (
    export_variable_projection_model_comparison,
    fit_global_observable_variable_projection_model_comparison,
)
from odefit.fitting.variable_projection_multistart import (
    export_variable_projection_multistart_summary,
    fit_global_observable_variable_projection_multistart,
)
from odefit.fitting.variable_projection_multistart_model_comparison import (
    export_variable_projection_multistart_model_comparison,
    fit_global_observable_variable_projection_multistart_model_comparison,
)
from odefit.fitting.variable_projection_profile_likelihood import (
    export_profile_likelihood_result,
    fit_variable_projection_profile_likelihood,
)
from odefit.model.model_spec import ModelSpec, build_model_spec
from odefit.model.ode_generator import generate_ode_lines
from odefit.performance.backend_capabilities import (
    build_backend_capabilities_table,
    summarize_backend_strategy,
)
from odefit.performance.benchmarking import (
    run_default_benchmarks,
    write_benchmark_results,
)
from odefit.plotting.timecourse_plots import (
    plot_simulation_timecourse,
    save_figure,
)
from odefit.simulation.simulation_result import SimulationResult
from odefit.simulation.simulation_settings import SimulationSettings
from odefit.simulation.solver import simulate_model


def read_model_file(model_path: str | Path) -> ModelSpec:
    """
    Read a model text file and build a ModelSpec.
    """

    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Model file does not exist: {path}")

    return build_model_spec(path.read_text())


def load_fit_config(config_path: str | Path | None) -> dict:
    """
    Load a JSON configuration file.

    Returns an empty dict if no config path is supplied.
    """

    if config_path is None:
        return {}

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r") as handle:
        return json.load(handle)


def get_config_value(
    args: argparse.Namespace,
    config: dict,
    argument_name: str,
    default=None,
    required: bool = False,
):
    """
    Get an argument value, preferring command-line value over config value.
    """

    cli_value = getattr(args, argument_name)

    if cli_value is not None:
        value = cli_value
    else:
        value = config.get(argument_name, default)

    if required and value is None:
        raise ValueError(f"Missing required argument/config value: {argument_name}")

    return value


def get_config_list_or_dict_value(
    args: argparse.Namespace,
    config: dict,
    argument_name: str,
    alternative_config_name: str | None = None,
):
    """
    Get a list/dict config value, preferring CLI over config.

    Supports alternative config keys like:
        parameter -> parameters
        initial -> initial_conditions
        signal_weight -> signal_weights
    """

    cli_value = getattr(args, argument_name)

    if cli_value is not None:
        return cli_value

    if argument_name in config:
        return config[argument_name]

    if alternative_config_name is not None and alternative_config_name in config:
        return config[alternative_config_name]

    return None


def parse_mapping_entries(
    mapping_entries: list[str] | dict[str, str] | None,
) -> dict[str, str]:
    """
    Parse mapping entries.

    CLI format:
        ["data_column:species"]

    JSON config format:
        {
            "data_column": "species"
        }
    """

    if not mapping_entries:
        return {}

    if isinstance(mapping_entries, dict):
        return {
            str(data_column): str(species)
            for data_column, species in mapping_entries.items()
        }

    mapping: dict[str, str] = {}

    for entry in mapping_entries:
        parts = entry.split(":")

        if len(parts) != 2:
            raise ValueError(
                f"Mapping entries must have format data_column:species. Got: {entry}"
            )

        data_column, species = parts

        if not data_column or not species:
            raise ValueError(f"Invalid mapping entry: {entry}")

        mapping[data_column] = species

    return mapping


def build_default_species_mapping(
    signal_columns: list[str],
    model: ModelSpec,
) -> dict[str, str]:
    """
    Build a default data-column to species mapping.

    Rules:
    - If there is one signal column and species A exists, map signal -> A.
    - Otherwise, map columns to species with identical names.
    """

    if len(signal_columns) == 1 and "A" in model.species:
        return {
            signal_columns[0]: "A",
        }

    mapping = {}

    for signal_column in signal_columns:
        if signal_column in model.species:
            mapping[signal_column] = signal_column

    missing_columns = set(signal_columns) - set(mapping)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            "Could not infer mapping for signal columns: "
            f"{missing}. Use --mapping data_column:species."
        )

    return mapping


def parse_parameter_entries(
    parameter_entries: list[str] | dict | None,
) -> dict[str, tuple[float, float, float]]:
    """
    Parse parameter entries.

    CLI format:
        ["k1f:0.01:0:10"]

    JSON config format:
        {
            "k1f": {
                "initial_guess": 0.01,
                "lower_bound": 0.0,
                "upper_bound": 10.0
            }
        }
    """

    if not parameter_entries:
        return {}

    parsed: dict[str, tuple[float, float, float]] = {}

    if isinstance(parameter_entries, dict):
        for name, values in parameter_entries.items():
            parsed[str(name)] = (
                float(values["initial_guess"]),
                float(values.get("lower_bound", 0.0)),
                float(values.get("upper_bound", 100.0)),
            )

        return parsed

    for entry in parameter_entries:
        parts = entry.split(":")

        if len(parts) != 4:
            raise ValueError(
                "Parameter entries must have format "
                "name:initial_guess:lower_bound:upper_bound. "
                f"Got: {entry}"
            )

        name, guess, lower, upper = parts

        parsed[name] = (
            float(guess),
            float(lower),
            float(upper),
        )

    return parsed


def build_parameter_specs(
    model: ModelSpec,
    parameter_entries: list[str] | dict | None = None,
    default_guess: float = 0.1,
    default_lower: float = 0.0,
    default_upper: float = 100.0,
) -> list[ParameterSpec]:
    """
    Build ParameterSpec objects for model parameters.

    CLI/config parameter entries override defaults.
    """

    overrides = parse_parameter_entries(parameter_entries)

    unknown_parameters = set(overrides) - set(model.parameters)

    if unknown_parameters:
        unknown = ", ".join(sorted(unknown_parameters))
        raise ValueError(f"Parameter override not present in model: {unknown}")

    parameter_specs = []

    for parameter_name in model.parameters:
        if parameter_name in overrides:
            guess, lower, upper = overrides[parameter_name]
        else:
            guess = default_guess
            lower = default_lower
            upper = default_upper

        parameter_specs.append(
            ParameterSpec(
                name=parameter_name,
                initial_guess=guess,
                lower_bound=lower,
                upper_bound=upper,
            )
        )

    return parameter_specs


def parse_initial_condition_entries(
    initial_entries: list[str] | dict | None,
) -> dict[str, tuple[float, bool, float, float]]:
    """
    Parse initial-condition entries.

    CLI format:
        ["A:1.0:fixed:0:10"]

    JSON config format:
        {
            "A": {
                "value": 1.0,
                "mode": "fixed",
                "lower_bound": 0.0,
                "upper_bound": 10.0
            }
        }
    """

    if not initial_entries:
        return {}

    parsed: dict[str, tuple[float, bool, float, float]] = {}

    if isinstance(initial_entries, dict):
        for species, values in initial_entries.items():
            mode = values.get("mode", "fixed")

            if mode not in {"fixed", "fit"}:
                raise ValueError(
                    f"Initial condition mode must be 'fixed' or 'fit'. Got: {mode}"
                )

            parsed[str(species)] = (
                float(values["value"]),
                mode == "fixed",
                float(values.get("lower_bound", 0.0)),
                float(values.get("upper_bound", 100.0)),
            )

        return parsed

    for entry in initial_entries:
        parts = entry.split(":")

        if len(parts) != 5:
            raise ValueError(
                "Initial condition entries must have format "
                "species:value:fixed_or_fit:lower_bound:upper_bound. "
                f"Got: {entry}"
            )

        species, value, mode, lower, upper = parts

        if mode not in {"fixed", "fit"}:
            raise ValueError(
                f"Initial condition mode must be 'fixed' or 'fit'. Got: {mode}"
            )

        parsed[species] = (
            float(value),
            mode == "fixed",
            float(lower),
            float(upper),
        )

    return parsed


def build_initial_condition_specs(
    model: ModelSpec,
    initial_entries: list[str] | dict | None = None,
) -> list[InitialConditionSpec]:
    """
    Build InitialConditionSpec objects.

    Defaults:
    - A starts fixed at 1.0 if A exists.
    - Otherwise, the first species starts fixed at 1.0.
    - Other species start fixed at 0.0.
    """

    overrides = parse_initial_condition_entries(initial_entries)

    unknown_species = set(overrides) - set(model.species)

    if unknown_species:
        unknown = ", ".join(sorted(unknown_species))
        raise ValueError(f"Initial condition species not present in model: {unknown}")

    primary_species = "A" if "A" in model.species else model.species[0]

    initial_condition_specs = []

    for species_name in model.species:
        if species_name in overrides:
            value, fixed, lower, upper = overrides[species_name]
        else:
            value = 1.0 if species_name == primary_species else 0.0
            fixed = True
            lower = 0.0
            upper = 100.0

        initial_condition_specs.append(
            InitialConditionSpec(
                species=species_name,
                initial_guess=value,
                lower_bound=lower,
                upper_bound=upper,
                fixed=fixed,
                fixed_value=value if fixed else None,
            )
        )

    return initial_condition_specs


def parse_signal_weight_entries(
    weight_entries: list[str] | dict[str, float] | None,
) -> dict[str, float] | None:
    """
    Parse signal weight entries.

    CLI format:
        ["amide:2.0"]

    JSON config format:
        {
            "amide": 2.0
        }
    """

    if not weight_entries:
        return None

    if isinstance(weight_entries, dict):
        return {
            str(data_column): float(weight)
            for data_column, weight in weight_entries.items()
        }

    weights: dict[str, float] = {}

    for entry in weight_entries:
        parts = entry.split(":")

        if len(parts) != 2:
            raise ValueError(
                "Signal weight entries must have format data_column:weight. "
                f"Got: {entry}"
            )

        data_column, weight = parts

        weights[data_column] = float(weight)

    return weights


def parse_float_value_entries(
    value_entries: list[str] | dict | None,
) -> dict[str, float]:
    """
    Parse name:value entries into a dictionary of floats.

    CLI format:
        ["k1f:0.5", "k1r:0.1"]

    JSON config format:
        {
            "k1f": 0.5,
            "k1r": 0.1
        }

    Also accepts config values like:
        {
            "A": {"value": 1.0}
        }
    """

    if not value_entries:
        return {}

    values: dict[str, float] = {}

    if isinstance(value_entries, dict):
        for name, value in value_entries.items():
            if isinstance(value, dict):
                if "value" in value:
                    values[str(name)] = float(value["value"])
                elif "initial_guess" in value:
                    values[str(name)] = float(value["initial_guess"])
                else:
                    raise ValueError(
                        f"Dictionary entry for {name} must contain 'value' "
                        "or 'initial_guess'"
                    )
            else:
                values[str(name)] = float(value)

        return values

    for entry in value_entries:
        parts = entry.split(":")

        if len(parts) != 2:
            raise ValueError(f"Value entries must have format name:value. Got: {entry}")

        name, value = parts

        if not name:
            raise ValueError(f"Invalid value entry: {entry}")

        values[name] = float(value)

    return values


def build_simulation_timepoints(
    timepoints: list[float] | list[str] | None = None,
    time_start: float | None = None,
    time_end: float | None = None,
    num_points: int | None = None,
) -> np.ndarray:
    """
    Build simulation timepoints.

    Either provide explicit timepoints, or provide:
        time_start, time_end, num_points
    """

    if timepoints is not None:
        return np.asarray(
            [float(timepoint) for timepoint in timepoints],
            dtype=float,
        )

    if time_start is None or time_end is None or num_points is None:
        raise ValueError(
            "Simulation requires either explicit timepoints or "
            "time_start, time_end, and num_points"
        )

    if num_points < 2:
        raise ValueError("num_points must be at least 2")

    return np.linspace(
        float(time_start),
        float(time_end),
        int(num_points),
    )


def build_simulation_dataframe(
    simulation_result: SimulationResult,
) -> pd.DataFrame:
    """
    Build a simulation output dataframe.

    Columns:
        time, species_1, species_2, ...
    """

    data = {
        "time": simulation_result.timepoints,
    }

    for species_name in simulation_result.species:
        data[species_name] = simulation_result.get_species_values(species_name)

    return pd.DataFrame(data)


def write_simulation_csv(
    simulation_result: SimulationResult,
    output_csv: str | Path,
) -> Path:
    """
    Write simulation result to CSV.
    """

    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = build_simulation_dataframe(simulation_result)
    dataframe.to_csv(path, index=False)

    return path


def command_generate_odes(args: argparse.Namespace) -> None:
    """
    Generate ODE text from a model file.
    """

    model = read_model_file(args.model)

    ode_lines = generate_ode_lines(model)

    print("Detected species:")
    for species in model.species:
        print(f"  {species}")

    print("\nDetected parameters:")
    for parameter in model.parameters:
        print(f"  {parameter}")

    print("\nGenerated ODEs:")
    for line in ode_lines:
        print(line)

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(ode_lines) + "\n")
        print(f"\nWrote ODEs to: {output_path}")


def command_simulate(args: argparse.Namespace) -> None:
    """
    Simulate a model from CLI/config and write simulated curves to CSV.
    """

    config = load_fit_config(args.config)

    model_path = get_config_value(
        args=args,
        config=config,
        argument_name="model",
        required=True,
    )

    model = read_model_file(model_path)

    parameter_entries = get_config_list_or_dict_value(
        args=args,
        config=config,
        argument_name="parameter_value",
        alternative_config_name="parameter_values",
    )

    initial_entries = get_config_list_or_dict_value(
        args=args,
        config=config,
        argument_name="initial_value",
        alternative_config_name="initial_values",
    )

    if parameter_entries is None:
        parameter_entries = config.get("parameters")

    if initial_entries is None:
        initial_entries = config.get("initial_conditions")

    parameters = parse_float_value_entries(parameter_entries)
    initial_conditions = parse_float_value_entries(initial_entries)

    timepoints = get_config_value(
        args=args,
        config=config,
        argument_name="timepoints",
        default=None,
    )

    time_start = get_config_value(
        args=args,
        config=config,
        argument_name="time_start",
        default=None,
    )

    time_end = get_config_value(
        args=args,
        config=config,
        argument_name="time_end",
        default=None,
    )

    num_points = get_config_value(
        args=args,
        config=config,
        argument_name="num_points",
        default=None,
    )

    simulation_timepoints = build_simulation_timepoints(
        timepoints=timepoints,
        time_start=time_start,
        time_end=time_end,
        num_points=num_points,
    )

    method = get_config_value(
        args=args,
        config=config,
        argument_name="method",
        default="LSODA",
    )

    rtol = get_config_value(
        args=args,
        config=config,
        argument_name="rtol",
        default=1e-6,
    )

    atol = get_config_value(
        args=args,
        config=config,
        argument_name="atol",
        default=1e-9,
    )

    clip_negative_concentrations = bool(
        config.get("clip_negative_concentrations", False)
    ) or bool(args.clip_negative_concentrations)

    warn_on_negative_values = bool(config.get("warn_on_negative_values", True))

    if args.no_negative_warnings:
        warn_on_negative_values = False

    settings = SimulationSettings(
        method=method,
        rtol=rtol,
        atol=atol,
        clip_negative_concentrations=clip_negative_concentrations,
        warn_on_negative_values=warn_on_negative_values,
    )

    result = simulate_model(
        model=model,
        parameters=parameters,
        initial_conditions=initial_conditions,
        timepoints=simulation_timepoints,
        settings=settings,
    )

    output_csv = get_config_value(
        args=args,
        config=config,
        argument_name="output_csv",
        required=True,
    )

    written_csv = write_simulation_csv(
        simulation_result=result,
        output_csv=output_csv,
    )

    output_plot = get_config_value(
        args=args,
        config=config,
        argument_name="output_plot",
        default=None,
    )

    written_plot = None

    if output_plot is not None:
        fig, _ = plot_simulation_timecourse(result)

        written_plot = save_figure(
            fig=fig,
            file_path=output_plot,
        )

    print("Simulation success:", result.success)
    print("Message:", result.message)

    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  {warning}")

    print(f"\nWrote simulation CSV to: {written_csv}")

    if written_plot is not None:
        print(f"Wrote simulation plot to: {written_plot}")


def command_performance_info(args: argparse.Namespace) -> None:
    """
    Print optional performance backend availability and acceleration roadmap.
    """

    table = build_backend_capabilities_table()

    print("Performance backend availability:")
    print(table.to_string(index=False))

    print("\nRecommended acceleration roadmap:")
    for line in summarize_backend_strategy():
        print(f"  {line}")


def command_benchmark_performance(args: argparse.Namespace) -> None:
    """
    Run small benchmark suite for current CPU/SciPy backend.
    """

    print("Running mODEler performance benchmarks...")
    print("This may take a little while.\n")

    results = run_default_benchmarks()

    table_path = write_benchmark_results(
        results=results,
        output_path=args.output,
    )

    print("Benchmark results:")
    for result in results:
        print(f"  {result.name}: {result.elapsed_seconds:.4f} s {result.metadata}")

    print(f"\nWrote benchmark table to: {table_path}")


def command_fit_amyloid_aggregation(args: argparse.Namespace) -> None:
    """
    Fit the built-in amyloid aggregation multi-condition workflow.
    """

    config = load_fit_config(args.config)

    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    output = fit_amyloid_aggregation_from_config(config)
    result = output["result"]

    print("Amyloid aggregation fit success:", result.success)
    print("Message:", result.message)

    print("\nFitted parameters:")
    for name, value in result.fitted_parameters.items():
        print(f"  {name}: {value:.8g}")

    print("\nFit statistics:")
    for name, value in result.statistics.items():
        print(f"  {name}: {value}")

    written_files = output.get("written_files", {})

    if written_files:
        print("\nWrote outputs:")
        for name, path in written_files.items():
            print(f"  {name}: {path}")


def get_peak_filtering_settings(
    args: argparse.Namespace,
    config: dict,
) -> dict:
    """
    Get HSQC/global observable peak-filtering settings from CLI/config.

    CLI values override config values.

    Filtering options:
    - max_missing_fraction
    - min_initial_intensity
    - initial_points
    - min_dynamic_range
    - interpolate_missing
    """

    max_missing_fraction = get_config_value(
        args=args,
        config=config,
        argument_name="max_missing_fraction",
        default=0.0,
    )

    min_initial_intensity = get_config_value(
        args=args,
        config=config,
        argument_name="min_initial_intensity",
        default=None,
    )

    initial_points = get_config_value(
        args=args,
        config=config,
        argument_name="initial_points",
        default=1,
    )

    min_dynamic_range = get_config_value(
        args=args,
        config=config,
        argument_name="min_dynamic_range",
        default=None,
    )

    interpolate_missing = bool(config.get("interpolate_missing", True))

    if getattr(args, "no_interpolate_missing", False):
        interpolate_missing = False

    return {
        "max_missing_fraction": float(max_missing_fraction),
        "min_initial_intensity": (
            None if min_initial_intensity is None else float(min_initial_intensity)
        ),
        "initial_points": int(initial_points),
        "min_dynamic_range": (
            None if min_dynamic_range is None else float(min_dynamic_range)
        ),
        "interpolate_missing": interpolate_missing,
    }


def write_peak_filtering_table(
    filtering_result,
    output_dir: str | Path,
) -> Path:
    """
    Write peak-filtering results to peak_filtering.csv.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    table = build_peak_filtering_table(filtering_result)

    path = output_path / "peak_filtering.csv"
    table.to_csv(path, index=False)

    return path


def build_model_specs_from_comparison_config(
    config: dict,
) -> dict[str, ModelSpec]:
    """
    Build named ModelSpec objects from a comparison config.

    Supported config formats:

    Inline model text:

        "model_texts": {
            "irreversible": "A>B",
            "reversible": "A-B"
        }

    or:

        "models": {
            "irreversible": "A>B",
            "reversible": "A-B"
        }

    Model files:

        "model_files": {
            "irreversible": "models/irreversible.txt",
            "reversible": "models/reversible.txt"
        }
    """

    model_texts = config.get("model_texts")

    if model_texts is None:
        model_texts = config.get("models")

    if model_texts is not None:
        if not isinstance(model_texts, dict):
            raise ValueError("model_texts/models must be a dictionary")

        return build_model_specs_from_texts(model_texts)

    model_files = config.get("model_files")

    if model_files is not None:
        if not isinstance(model_files, dict):
            raise ValueError("model_files must be a dictionary")

        models: dict[str, ModelSpec] = {}

        for model_name, model_file in model_files.items():
            path = Path(model_file)

            if not path.exists():
                raise FileNotFoundError(f"Model file does not exist: {path}")

            models[str(model_name)] = build_model_spec(
                text=path.read_text(),
                name=str(model_name),
            )

        return models

    raise ValueError(
        "Global observable model comparison requires one of: "
        "model_texts, models, or model_files"
    )


def build_parameter_specs_by_model_from_config(
    models: dict[str, ModelSpec],
    config: dict,
    default_guess: float = 0.1,
    default_lower: float = 0.0,
    default_upper: float = 100.0,
) -> dict[str, list[ParameterSpec]]:
    """
    Build parameter specs for each comparison model.

    Preferred config format:

        "parameters_by_model": {
            "irreversible": {
                "k1f": {
                    "initial_guess": 0.1,
                    "lower_bound": 0.001,
                    "upper_bound": 10.0
                }
            },
            "reversible": {
                "k1f": {...},
                "k1r": {...}
            }
        }

    Missing model entries fall back to default guesses/bounds.
    """

    parameters_by_model = config.get("parameters_by_model", {})

    if parameters_by_model is None:
        parameters_by_model = {}

    if not isinstance(parameters_by_model, dict):
        raise ValueError("parameters_by_model must be a dictionary")

    specs_by_model: dict[str, list[ParameterSpec]] = {}

    for model_name, model in models.items():
        parameter_entries = parameters_by_model.get(model_name)

        specs_by_model[model_name] = build_parameter_specs(
            model=model,
            parameter_entries=parameter_entries,
            default_guess=default_guess,
            default_lower=default_lower,
            default_upper=default_upper,
        )

    return specs_by_model


def build_initial_condition_specs_by_model_from_config(
    models: dict[str, ModelSpec],
    config: dict,
) -> dict[str, list[InitialConditionSpec]]:
    """
    Build initial condition specs for each comparison model.

    Preferred config format:

        "initial_conditions_by_model": {
            "irreversible": {
                "A": {"value": 1.0, "mode": "fixed"},
                "B": {"value": 0.0, "mode": "fixed"}
            },
            "reversible": {
                "A": {"value": 1.0, "mode": "fixed"},
                "B": {"value": 0.0, "mode": "fixed"}
            }
        }

    Missing model entries use build_initial_condition_specs defaults.
    """

    initial_conditions_by_model = config.get("initial_conditions_by_model", {})

    if initial_conditions_by_model is None:
        initial_conditions_by_model = {}

    if not isinstance(initial_conditions_by_model, dict):
        raise ValueError("initial_conditions_by_model must be a dictionary")

    specs_by_model: dict[str, list[InitialConditionSpec]] = {}

    for model_name, model in models.items():
        initial_entries = initial_conditions_by_model.get(model_name)

        specs_by_model[model_name] = build_initial_condition_specs(
            model=model,
            initial_entries=initial_entries,
        )

    return specs_by_model


def command_fit(args: argparse.Namespace) -> None:
    """
    Fit a model to a CSV dataset using direct species mapping.

    Values can come from CLI arguments or from a JSON config file.
    CLI arguments override config-file values.
    """

    config = load_fit_config(args.config)

    model_path = get_config_value(
        args=args,
        config=config,
        argument_name="model",
        required=True,
    )

    data_path = get_config_value(
        args=args,
        config=config,
        argument_name="data",
        required=True,
    )

    time_column = get_config_value(
        args=args,
        config=config,
        argument_name="time_column",
        default="time",
    )

    signal_columns = get_config_value(
        args=args,
        config=config,
        argument_name="signal_columns",
        required=True,
    )

    output_dir = get_config_value(
        args=args,
        config=config,
        argument_name="output_dir",
        required=True,
    )

    model = read_model_file(model_path)

    dataframe = pd.read_csv(data_path)

    dataset = Dataset(
        raw_dataframe=dataframe,
        time_column=time_column,
        signal_columns=signal_columns,
    )

    mapping_entries = get_config_list_or_dict_value(
        args=args,
        config=config,
        argument_name="mapping",
    )

    species_mapping = parse_mapping_entries(mapping_entries)

    if not species_mapping:
        species_mapping = build_default_species_mapping(
            signal_columns=signal_columns,
            model=model,
        )

    parameter_entries = get_config_list_or_dict_value(
        args=args,
        config=config,
        argument_name="parameter",
        alternative_config_name="parameters",
    )

    initial_entries = get_config_list_or_dict_value(
        args=args,
        config=config,
        argument_name="initial",
        alternative_config_name="initial_conditions",
    )

    signal_weight_entries = get_config_list_or_dict_value(
        args=args,
        config=config,
        argument_name="signal_weight",
        alternative_config_name="signal_weights",
    )

    default_parameter_guess = get_config_value(
        args=args,
        config=config,
        argument_name="default_parameter_guess",
        default=0.1,
    )

    default_parameter_lower = get_config_value(
        args=args,
        config=config,
        argument_name="default_parameter_lower",
        default=0.0,
    )

    default_parameter_upper = get_config_value(
        args=args,
        config=config,
        argument_name="default_parameter_upper",
        default=100.0,
    )

    method = get_config_value(
        args=args,
        config=config,
        argument_name="method",
        default="trf",
    )

    loss = get_config_value(
        args=args,
        config=config,
        argument_name="loss",
        default="linear",
    )

    max_nfev = get_config_value(
        args=args,
        config=config,
        argument_name="max_nfev",
        default=None,
    )

    rtol = get_config_value(
        args=args,
        config=config,
        argument_name="rtol",
        default=1e-6,
    )

    atol = get_config_value(
        args=args,
        config=config,
        argument_name="atol",
        default=1e-9,
    )

    no_plots = bool(config.get("no_plots", False)) or bool(args.no_plots)

    parameter_specs = build_parameter_specs(
        model=model,
        parameter_entries=parameter_entries,
        default_guess=default_parameter_guess,
        default_lower=default_parameter_lower,
        default_upper=default_parameter_upper,
    )

    initial_condition_specs = build_initial_condition_specs(
        model=model,
        initial_entries=initial_entries,
    )

    settings = FitSettings(
        species_mapping=species_mapping,
        use_normalized_data=False,
        method=method,
        loss=loss,
        max_nfev=max_nfev,
        rtol=rtol,
        atol=atol,
        signal_weights=parse_signal_weight_entries(signal_weight_entries),
    )

    result = fit_model(
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        settings=settings,
    )

    written_files = export_fit_bundle(
        fit_result=result,
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        species_mapping=species_mapping,
        include_plots=not no_plots,
        fit_settings=settings,
        command="fit",
        config_path=args.config,
    )

    print("Fit success:", result.success)
    print("Message:", result.message)
    print("Fitted parameters:", result.fitted_parameters)
    print("Fitted initial conditions:", result.fitted_initial_conditions)
    print("Statistics:", result.statistics)
    print(f"\nWrote output bundle to: {Path(output_dir)}")

    print("\nWritten files:")
    for name, path in written_files.items():
        print(f"  {name}: {path}")


def command_multistart(args: argparse.Namespace) -> None:
    """
    Run multistart fitting from CLI/config.

    This currently supports direct species mapping.
    """

    config = load_fit_config(args.config)

    model_path = get_config_value(
        args=args,
        config=config,
        argument_name="model",
        required=True,
    )

    data_path = get_config_value(
        args=args,
        config=config,
        argument_name="data",
        required=True,
    )

    time_column = get_config_value(
        args=args,
        config=config,
        argument_name="time_column",
        default="time",
    )

    signal_columns = get_config_value(
        args=args,
        config=config,
        argument_name="signal_columns",
        required=True,
    )

    output_dir = get_config_value(
        args=args,
        config=config,
        argument_name="output_dir",
        required=True,
    )

    n_starts = int(
        get_config_value(
            args=args,
            config=config,
            argument_name="n_starts",
            default=10,
        )
    )

    n_workers = get_config_value(
        args=args,
        config=config,
        argument_name="n_workers",
        default=1,
    )

    if n_workers is not None:
        n_workers = int(n_workers)

    random_seed = get_config_value(
        args=args,
        config=config,
        argument_name="random_seed",
        default=None,
    )

    if random_seed is not None:
        random_seed = int(random_seed)

    sort_by = get_config_value(
        args=args,
        config=config,
        argument_name="sort_by",
        default="aic",
    )

    log_uniform = bool(config.get("log_uniform", True))

    if args.linear_sampling:
        log_uniform = False

    no_plots = bool(config.get("no_plots", False)) or bool(args.no_plots)

    model = read_model_file(model_path)

    dataframe = pd.read_csv(data_path)

    dataset = Dataset(
        raw_dataframe=dataframe,
        time_column=time_column,
        signal_columns=signal_columns,
    )

    mapping_entries = get_config_list_or_dict_value(
        args=args,
        config=config,
        argument_name="mapping",
    )

    species_mapping = parse_mapping_entries(mapping_entries)

    if not species_mapping:
        species_mapping = build_default_species_mapping(
            signal_columns=signal_columns,
            model=model,
        )

    parameter_entries = get_config_list_or_dict_value(
        args=args,
        config=config,
        argument_name="parameter",
        alternative_config_name="parameters",
    )

    initial_entries = get_config_list_or_dict_value(
        args=args,
        config=config,
        argument_name="initial",
        alternative_config_name="initial_conditions",
    )

    signal_weight_entries = get_config_list_or_dict_value(
        args=args,
        config=config,
        argument_name="signal_weight",
        alternative_config_name="signal_weights",
    )

    default_parameter_guess = get_config_value(
        args=args,
        config=config,
        argument_name="default_parameter_guess",
        default=0.1,
    )

    default_parameter_lower = get_config_value(
        args=args,
        config=config,
        argument_name="default_parameter_lower",
        default=0.0,
    )

    default_parameter_upper = get_config_value(
        args=args,
        config=config,
        argument_name="default_parameter_upper",
        default=100.0,
    )

    method = get_config_value(
        args=args,
        config=config,
        argument_name="method",
        default="trf",
    )

    loss = get_config_value(
        args=args,
        config=config,
        argument_name="loss",
        default="linear",
    )

    max_nfev = get_config_value(
        args=args,
        config=config,
        argument_name="max_nfev",
        default=None,
    )

    rtol = get_config_value(
        args=args,
        config=config,
        argument_name="rtol",
        default=1e-6,
    )

    atol = get_config_value(
        args=args,
        config=config,
        argument_name="atol",
        default=1e-9,
    )

    parameter_specs = build_parameter_specs(
        model=model,
        parameter_entries=parameter_entries,
        default_guess=default_parameter_guess,
        default_lower=default_parameter_lower,
        default_upper=default_parameter_upper,
    )

    initial_condition_specs = build_initial_condition_specs(
        model=model,
        initial_entries=initial_entries,
    )

    settings = FitSettings(
        species_mapping=species_mapping,
        use_normalized_data=False,
        method=method,
        loss=loss,
        max_nfev=max_nfev,
        rtol=rtol,
        atol=atol,
        signal_weights=parse_signal_weight_entries(signal_weight_entries),
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if n_workers is None or n_workers <= 1:
        print(f"Running serial multistart with {n_starts} starts")

        multistart_result = fit_multistart(
            model=model,
            dataset=dataset,
            parameter_specs=parameter_specs,
            initial_condition_specs=initial_condition_specs,
            settings=settings,
            n_starts=n_starts,
            random_seed=random_seed,
            sort_by=sort_by,
            log_uniform=log_uniform,
        )

        comparison_path = export_multistart_comparison(
            multistart_result=multistart_result,
            file_path=output_path / "multistart_comparison.csv",
        )

        written_summary_files = {
            "multistart_comparison": comparison_path,
        }

        best_result = multistart_result.best_result
        best_index = multistart_result.best_index

    else:
        print(
            f"Running parallel multistart with {n_starts} starts "
            f"using {n_workers} workers"
        )

        parallel_result = fit_multistart_parallel(
            model=model,
            dataset=dataset,
            parameter_specs=parameter_specs,
            initial_condition_specs=initial_condition_specs,
            settings=settings,
            n_starts=n_starts,
            n_workers=n_workers,
            random_seed=random_seed,
            sort_by=sort_by,
            log_uniform=log_uniform,
        )

        written_summary_files = export_parallel_multistart_summary(
            parallel_result=parallel_result,
            output_dir=output_path,
        )

        multistart_result = parallel_result.successful_result
        best_result = multistart_result.best_result
        best_index = multistart_result.best_index

        print("Submitted starts:", parallel_result.n_submitted)
        print("Successful starts:", parallel_result.n_successful)
        print("Failed starts:", parallel_result.n_failed)

    starting_parameters_table = pd.DataFrame(multistart_result.starting_parameter_sets)

    starting_parameters_table.insert(
        0,
        "start_index",
        list(range(len(starting_parameters_table))),
    )

    starting_parameters_path = output_path / "multistart_starting_parameters.csv"
    starting_parameters_table.to_csv(starting_parameters_path, index=False)

    written_summary_files["multistart_starting_parameters"] = starting_parameters_path

    best_fit_output_dir = output_path / "best_fit"

    best_fit_files = export_fit_bundle(
        fit_result=best_result,
        model=model,
        dataset=dataset,
        output_dir=best_fit_output_dir,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        species_mapping=species_mapping,
        include_plots=not no_plots,
        fit_settings=settings,
        command="multistart-best-fit",
        config_path=args.config,
        extra_run_metadata={
            "best_start_index": best_index,
            "n_starts": n_starts,
            "n_workers": n_workers,
            "sort_by": sort_by,
        },
    )

    print("\nBest start index:", best_index)
    print("Best fit success:", best_result.success)
    print("Best fit message:", best_result.message)
    print("Best fitted parameters:", best_result.fitted_parameters)
    print("Best statistics:", best_result.statistics)

    print(f"\nWrote multistart outputs to: {output_path}")
    print(f"Wrote best fit bundle to: {best_fit_output_dir}")

    print("\nWritten summary files:")
    for name, path in written_summary_files.items():
        print(f"  {name}: {path}")

    print("\nWritten best-fit files:")
    for name, path in best_fit_files.items():
        print(f"  {name}: {path}")


def command_fit_global_observables(args: argparse.Namespace) -> None:
    """
    Fit many observable columns to one shared global kinetic model.

    Intended first use case:
        wide-format assigned HSQC peak intensities

    Example data:

        time,A23_HN,G45_HN,L78_HN
        0,1000,850,1200
        1,920,810,1105

    Default observable model:

        signal_i(t) = scale_i * observed_species(t) + offset_i
    """

    config = load_fit_config(args.config)
    engine_name = str(config.get("engine_name", "reference"))

    model_path = get_config_value(
        args=args,
        config=config,
        argument_name="model",
        required=True,
    )

    data_path = get_config_value(
        args=args,
        config=config,
        argument_name="data",
        required=True,
    )

    time_column = get_config_value(
        args=args,
        config=config,
        argument_name="time_column",
        default="time",
    )

    signal_columns = get_config_value(
        args=args,
        config=config,
        argument_name="signal_columns",
        default=None,
    )

    exclude_columns = get_config_value(
        args=args,
        config=config,
        argument_name="exclude_columns",
        default=None,
    )

    observed_species = get_config_value(
        args=args,
        config=config,
        argument_name="observed_species",
        default="A",
    )

    output_dir = get_config_value(
        args=args,
        config=config,
        argument_name="output_dir",
        required=True,
    )

    model = read_model_file(model_path)

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

    dataset, filtering_result = read_wide_observable_dataset_with_filtering(
        file_path=data_path,
        time_column=time_column,
        signal_columns=signal_columns,
        exclude_columns=exclude_columns,
        max_missing_fraction=peak_filtering_settings["max_missing_fraction"],
        min_initial_intensity=peak_filtering_settings["min_initial_intensity"],
        initial_points=peak_filtering_settings["initial_points"],
        min_dynamic_range=peak_filtering_settings["min_dynamic_range"],
        interpolate_missing=peak_filtering_settings["interpolate_missing"],
    )

    parameter_entries = get_config_list_or_dict_value(
        args=args,
        config=config,
        argument_name="parameter",
        alternative_config_name="parameters",
    )

    initial_entries = get_config_list_or_dict_value(
        args=args,
        config=config,
        argument_name="initial",
        alternative_config_name="initial_conditions",
    )

    signal_weight_entries = get_config_list_or_dict_value(
        args=args,
        config=config,
        argument_name="signal_weight",
        alternative_config_name="signal_weights",
    )

    default_parameter_guess = get_config_value(
        args=args,
        config=config,
        argument_name="default_parameter_guess",
        default=0.1,
    )

    default_parameter_lower = get_config_value(
        args=args,
        config=config,
        argument_name="default_parameter_lower",
        default=0.0,
    )

    default_parameter_upper = get_config_value(
        args=args,
        config=config,
        argument_name="default_parameter_upper",
        default=100.0,
    )

    method = get_config_value(
        args=args,
        config=config,
        argument_name="method",
        default="trf",
    )

    loss = get_config_value(
        args=args,
        config=config,
        argument_name="loss",
        default="linear",
    )

    max_nfev = get_config_value(
        args=args,
        config=config,
        argument_name="max_nfev",
        default=None,
    )

    rtol = get_config_value(
        args=args,
        config=config,
        argument_name="rtol",
        default=1e-6,
    )

    atol = get_config_value(
        args=args,
        config=config,
        argument_name="atol",
        default=1e-9,
    )

    fit_scale = bool(config.get("fit_scale", True))
    fit_offset = bool(config.get("fit_offset", True))

    scale_initial_guess = float(config.get("scale_initial_guess", 1.0))
    scale_lower_bound = float(config.get("scale_lower_bound", 0.0))
    scale_upper_bound = float(config.get("scale_upper_bound", float("inf")))

    offset_initial_guess = float(config.get("offset_initial_guess", 0.0))
    offset_lower_bound = float(config.get("offset_lower_bound", -float("inf")))
    offset_upper_bound = float(config.get("offset_upper_bound", float("inf")))

    no_plots = bool(config.get("no_plots", False)) or bool(args.no_plots)

    parameter_specs = build_parameter_specs(
        model=model,
        parameter_entries=parameter_entries,
        default_guess=default_parameter_guess,
        default_lower=default_parameter_lower,
        default_upper=default_parameter_upper,
    )

    initial_condition_specs = build_initial_condition_specs(
        model=model,
        initial_entries=initial_entries,
    )

    settings = FitSettings(
        species_mapping={},
        use_normalized_data=False,
        method=method,
        loss=loss,
        max_nfev=max_nfev,
        rtol=rtol,
        atol=atol,
        signal_weights=parse_signal_weight_entries(signal_weight_entries),
    )

    use_variable_projection = bool(
        config.get("use_variable_projection", False)
    ) or bool(getattr(args, "variable_projection", False))

    use_multispecies_variable_projection = bool(
        config.get("use_multispecies_variable_projection", False)
    )

    use_multistart = bool(config.get("use_multistart", False))

    variable_projection_backend = str(
        config.get("variable_projection_backend", "numpy")
    )

    variable_projection_method = str(config.get("variable_projection_method", "LSODA"))

    if use_multispecies_variable_projection:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if not isinstance(observed_species, list):
            raise ValueError(
                "Multispecies variable projection requires "
                "observed_species to be a list, e.g. ['A', 'B']."
            )

        print("Running global observable fit with multispecies variable projection")
        print(f"Observed species: {observed_species}")
        print(f"Observable columns: {len(dataset.signal_columns)}")
        print(f"Backend: {variable_projection_backend}")
        print(f"ODE method: {variable_projection_method}")

        if use_multistart:
            n_starts = int(config.get("n_starts", 10))
            random_seed = config.get("random_seed")

            if random_seed is not None:
                random_seed = int(random_seed)

            sort_by = config.get("sort_by", "bic")
            log_uniform = bool(config.get("log_uniform", True))

            result = (
                fit_global_observable_model_multispecies_variable_projection_multistart(
                    model=model,
                    dataset=dataset,
                    parameter_specs=parameter_specs,
                    initial_condition_specs=initial_condition_specs,
                    observed_species=observed_species,
                    settings=settings,
                    signal_columns=dataset.signal_columns,
                    fit_offset=fit_offset,
                    backend=variable_projection_backend,
                    method=variable_projection_method,
                    n_starts=n_starts,
                    random_seed=random_seed,
                    sort_by=sort_by,
                    log_uniform=log_uniform,
                    show_progress=True,
                    engine_name=engine_name,
                )
            )

            written_files = export_multispecies_variable_projection_multistart_summary(
                result=result,
                output_dir=output_path,
                export_best_fit=True,
            )

            peak_filtering_path = write_peak_filtering_table(
                filtering_result=filtering_result,
                output_dir=output_path,
            )

            written_files["peak_filtering"] = peak_filtering_path

            print("\nBest start:", result.best_index)
            print("Best fit success:", result.best_result.success)
            print("Best fit message:", result.best_result.message)
            print(
                "Best fitted kinetic parameters:", result.best_result.fitted_parameters
            )
            print("Best statistics:", result.best_result.statistics)

            print(
                "\nWrote multispecies variable projection multistart outputs "
                f"to: {output_path}"
            )

            print("\nWritten files:")
            for name, path in written_files.items():
                print(f"  {name}: {path}")

            return

        result = fit_global_observable_model_multispecies_variable_projection(
            model=model,
            dataset=dataset,
            parameter_specs=parameter_specs,
            initial_condition_specs=initial_condition_specs,
            observed_species=observed_species,
            settings=settings,
            signal_columns=dataset.signal_columns,
            fit_offset=fit_offset,
            backend=variable_projection_backend,
            method=variable_projection_method,
            engine_name=engine_name,
        )

        written_files = export_multispecies_variable_projection_fit(
            result=result,
            output_dir=output_path,
        )

        peak_filtering_path = write_peak_filtering_table(
            filtering_result=filtering_result,
            output_dir=output_path,
        )

        written_files["peak_filtering"] = peak_filtering_path

        print("\nMultispecies variable projection fit success:", result.success)
        print("Message:", result.message)
        print("Fitted kinetic parameters:", result.fitted_parameters)
        print("Statistics:", result.statistics)

        print(f"\nWrote multispecies variable projection outputs to: {output_path}")
        print("\nWritten files:")

        for name, path in written_files.items():
            print(f"  {name}: {path}")

        return

    if use_variable_projection:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("Running global observable fit with variable projection")
        print(f"Observed species: {observed_species}")
        print(f"Observable columns: {len(dataset.signal_columns)}")
        print(f"Backend: {variable_projection_backend}")
        print(f"ODE method: {variable_projection_method}")

        result = fit_global_observable_model_variable_projection(
            model=model,
            dataset=dataset,
            parameter_specs=parameter_specs,
            initial_condition_specs=initial_condition_specs,
            observed_species=observed_species,
            settings=settings,
            signal_columns=dataset.signal_columns,
            fit_scale=fit_scale,
            fit_offset=fit_offset,
            backend=variable_projection_backend,
            method=variable_projection_method,
            engine_name=engine_name,
        )

        written_files = export_variable_projection_fit(
            result=result,
            output_dir=str(output_path),
        )

        peak_filtering_path = write_peak_filtering_table(
            filtering_result=filtering_result,
            output_dir=output_path,
        )

        written_files["peak_filtering"] = str(peak_filtering_path)

        print("\nVariable projection fit success:", result.success)
        print("Message:", result.message)
        print("Fitted kinetic parameters:", result.fitted_parameters)
        print("Statistics:", result.statistics)

        print(f"\nWrote variable projection outputs to: {output_path}")
        print("\nWritten files:")

        for name, path in written_files.items():
            print(f"  {name}: {path}")

        return

    output = fit_global_observable_model(
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        observed_species=observed_species,
        settings=settings,
        signal_columns=dataset.signal_columns,
        fit_scale=fit_scale,
        fit_offset=fit_offset,
        scale_initial_guess=scale_initial_guess,
        scale_lower_bound=scale_lower_bound,
        scale_upper_bound=scale_upper_bound,
        offset_initial_guess=offset_initial_guess,
        offset_lower_bound=offset_lower_bound,
        offset_upper_bound=offset_upper_bound,
    )

    result = output.fit_result

    written_files = export_fit_bundle(
        fit_result=result,
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        observable_specs=output.observable_specs,
        species_mapping={},
        include_plots=not no_plots,
        fit_settings=settings,
        command="fit-global-observables",
        config_path=args.config,
        extra_run_metadata={
            "observed_species": observed_species,
            "n_observable_columns": len(dataset.signal_columns),
        },
    )

    peak_filtering_path = write_peak_filtering_table(
        filtering_result=filtering_result,
        output_dir=output_dir,
    )

    written_files["peak_filtering"] = peak_filtering_path

    print("Global observable fit success:", result.success)
    print("Message:", result.message)
    print("Observed species:", observed_species)
    print("Number of observable columns:", len(dataset.signal_columns))
    print("Kept observable columns:", len(filtering_result.kept_columns))
    print("Removed observable columns:", len(filtering_result.removed_columns))
    print("Fitted kinetic parameters:", result.fitted_parameters)
    print("Statistics:", result.statistics)
    print(f"\nWrote output bundle to: {Path(output_dir)}")

    print("\nWritten files:")
    for name, path in written_files.items():
        print(f"  {name}: {path}")


def command_multistart_global_observables(args: argparse.Namespace) -> None:
    """
    Run multistart fitting for many observable columns sharing one kinetic model.

    Intended first use case:
        wide-format assigned HSQC peak intensities

    Example data:

        time,A23_HN,G45_HN,L78_HN
        0,1000,850,1200
        1,920,810,1105

    Default observable model:

        signal_i(t) = scale_i * observed_species(t) + offset_i

    Shared globally:
        kinetic parameters
        initial conditions
        simulated species timecourses

    Signal-specific:
        scale_i
        offset_i
    """

    config = load_fit_config(args.config)

    use_variable_projection = bool(
        config.get("use_variable_projection", False)
    ) or bool(getattr(args, "variable_projection", False))

    if use_variable_projection:
        command_multistart_global_observables_variable_projection(
            args=args,
            config=config,
        )
        return

    model_path = get_config_value(
        args=args,
        config=config,
        argument_name="model",
        required=True,
    )

    data_path = get_config_value(
        args=args,
        config=config,
        argument_name="data",
        required=True,
    )

    time_column = get_config_value(
        args=args,
        config=config,
        argument_name="time_column",
        default="time",
    )

    signal_columns = get_config_value(
        args=args,
        config=config,
        argument_name="signal_columns",
        default=None,
    )

    exclude_columns = get_config_value(
        args=args,
        config=config,
        argument_name="exclude_columns",
        default=None,
    )

    observed_species = get_config_value(
        args=args,
        config=config,
        argument_name="observed_species",
        default="A",
    )

    output_dir = get_config_value(
        args=args,
        config=config,
        argument_name="output_dir",
        required=True,
    )

    n_starts = int(
        get_config_value(
            args=args,
            config=config,
            argument_name="n_starts",
            default=10,
        )
    )

    n_workers = get_config_value(
        args=args,
        config=config,
        argument_name="n_workers",
        default=1,
    )

    if n_workers is not None:
        n_workers = int(n_workers)

    random_seed = get_config_value(
        args=args,
        config=config,
        argument_name="random_seed",
        default=None,
    )

    if random_seed is not None:
        random_seed = int(random_seed)

    sort_by = get_config_value(
        args=args,
        config=config,
        argument_name="sort_by",
        default="aic",
    )

    log_uniform = bool(config.get("log_uniform", True))

    if args.linear_sampling:
        log_uniform = False

    model = read_model_file(model_path)

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

    dataset, filtering_result = read_wide_observable_dataset_with_filtering(
        file_path=data_path,
        time_column=time_column,
        signal_columns=signal_columns,
        exclude_columns=exclude_columns,
        max_missing_fraction=peak_filtering_settings["max_missing_fraction"],
        min_initial_intensity=peak_filtering_settings["min_initial_intensity"],
        initial_points=peak_filtering_settings["initial_points"],
        min_dynamic_range=peak_filtering_settings["min_dynamic_range"],
        interpolate_missing=peak_filtering_settings["interpolate_missing"],
    )

    parameter_entries = get_config_list_or_dict_value(
        args=args,
        config=config,
        argument_name="parameter",
        alternative_config_name="parameters",
    )

    initial_entries = get_config_list_or_dict_value(
        args=args,
        config=config,
        argument_name="initial",
        alternative_config_name="initial_conditions",
    )

    signal_weight_entries = get_config_list_or_dict_value(
        args=args,
        config=config,
        argument_name="signal_weight",
        alternative_config_name="signal_weights",
    )

    default_parameter_guess = get_config_value(
        args=args,
        config=config,
        argument_name="default_parameter_guess",
        default=0.1,
    )

    default_parameter_lower = get_config_value(
        args=args,
        config=config,
        argument_name="default_parameter_lower",
        default=0.0,
    )

    default_parameter_upper = get_config_value(
        args=args,
        config=config,
        argument_name="default_parameter_upper",
        default=100.0,
    )

    method = get_config_value(
        args=args,
        config=config,
        argument_name="method",
        default="trf",
    )

    loss = get_config_value(
        args=args,
        config=config,
        argument_name="loss",
        default="linear",
    )

    max_nfev = get_config_value(
        args=args,
        config=config,
        argument_name="max_nfev",
        default=None,
    )

    rtol = get_config_value(
        args=args,
        config=config,
        argument_name="rtol",
        default=1e-6,
    )

    atol = get_config_value(
        args=args,
        config=config,
        argument_name="atol",
        default=1e-9,
    )

    fit_scale = bool(config.get("fit_scale", True))
    fit_offset = bool(config.get("fit_offset", True))

    scale_initial_guess = float(config.get("scale_initial_guess", 1.0))
    scale_lower_bound = float(config.get("scale_lower_bound", 0.0))
    scale_upper_bound = float(config.get("scale_upper_bound", float("inf")))

    offset_initial_guess = float(config.get("offset_initial_guess", 0.0))
    offset_lower_bound = float(config.get("offset_lower_bound", -float("inf")))
    offset_upper_bound = float(config.get("offset_upper_bound", float("inf")))

    no_plots = bool(config.get("no_plots", False)) or bool(args.no_plots)

    parameter_specs = build_parameter_specs(
        model=model,
        parameter_entries=parameter_entries,
        default_guess=default_parameter_guess,
        default_lower=default_parameter_lower,
        default_upper=default_parameter_upper,
    )

    initial_condition_specs = build_initial_condition_specs(
        model=model,
        initial_entries=initial_entries,
    )

    observable_specs = build_shared_species_observable_specs(
        signal_columns=dataset.signal_columns,
        species=observed_species,
        fit_scale=fit_scale,
        fit_offset=fit_offset,
        scale_initial_guess=scale_initial_guess,
        scale_lower_bound=scale_lower_bound,
        scale_upper_bound=scale_upper_bound,
        offset_initial_guess=offset_initial_guess,
        offset_lower_bound=offset_lower_bound,
        offset_upper_bound=offset_upper_bound,
    )

    settings = FitSettings(
        species_mapping={},
        use_normalized_data=False,
        method=method,
        loss=loss,
        max_nfev=max_nfev,
        rtol=rtol,
        atol=atol,
        signal_weights=parse_signal_weight_entries(signal_weight_entries),
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    peak_filtering_path = write_peak_filtering_table(
        filtering_result=filtering_result,
        output_dir=output_path,
    )

    if n_workers is None or n_workers <= 1:
        print(f"Running serial global observable multistart with {n_starts} starts")

        multistart_result = fit_multistart(
            model=model,
            dataset=dataset,
            parameter_specs=parameter_specs,
            initial_condition_specs=initial_condition_specs,
            observable_specs=observable_specs,
            settings=settings,
            n_starts=n_starts,
            random_seed=random_seed,
            sort_by=sort_by,
            log_uniform=log_uniform,
        )

        comparison_path = export_multistart_comparison(
            multistart_result=multistart_result,
            file_path=output_path / "global_observable_multistart_comparison.csv",
        )

        written_summary_files = {
            "global_observable_multistart_comparison": comparison_path,
        }

        best_result = multistart_result.best_result
        best_index = multistart_result.best_index

    else:
        print(
            f"Running parallel global observable multistart with "
            f"{n_starts} starts using {n_workers} workers"
        )

        parallel_result = fit_multistart_parallel(
            model=model,
            dataset=dataset,
            parameter_specs=parameter_specs,
            initial_condition_specs=initial_condition_specs,
            observable_specs=observable_specs,
            settings=settings,
            n_starts=n_starts,
            n_workers=n_workers,
            random_seed=random_seed,
            sort_by=sort_by,
            log_uniform=log_uniform,
        )

        written_summary_files = export_parallel_multistart_summary(
            parallel_result=parallel_result,
            output_dir=output_path,
        )
        written_summary_files["peak_filtering"] = peak_filtering_path

        multistart_result = parallel_result.successful_result
        best_result = multistart_result.best_result
        best_index = multistart_result.best_index

        print("Submitted starts:", parallel_result.n_submitted)
        print("Successful starts:", parallel_result.n_successful)
        print("Failed starts:", parallel_result.n_failed)

    starting_parameters_table = pd.DataFrame(multistart_result.starting_parameter_sets)

    starting_parameters_table.insert(
        0,
        "start_index",
        list(range(len(starting_parameters_table))),
    )

    starting_parameters_path = (
        output_path / "global_observable_multistart_starting_parameters.csv"
    )

    starting_parameters_table.to_csv(
        starting_parameters_path,
        index=False,
    )

    written_summary_files["global_observable_multistart_starting_parameters"] = (
        starting_parameters_path
    )

    best_fit_output_dir = output_path / "best_fit"

    best_fit_files = export_fit_bundle(
        fit_result=best_result,
        model=model,
        dataset=dataset,
        output_dir=best_fit_output_dir,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        observable_specs=observable_specs,
        species_mapping={},
        include_plots=not no_plots,
        fit_settings=settings,
        command="multistart-global-observables-best-fit",
        config_path=args.config,
        extra_run_metadata={
            "observed_species": observed_species,
            "n_observable_columns": len(dataset.signal_columns),
            "best_start_index": best_index,
            "n_starts": n_starts,
            "n_workers": n_workers,
            "sort_by": sort_by,
        },
    )

    print("\nBest start index:", best_index)
    print("Best fit success:", best_result.success)
    print("Best fit message:", best_result.message)
    print("Observed species:", observed_species)
    print("Number of observable columns:", len(dataset.signal_columns))
    print("Kept observable columns:", len(filtering_result.kept_columns))
    print("Removed observable columns:", len(filtering_result.removed_columns))
    print("Best fitted kinetic parameters:", best_result.fitted_parameters)
    print("Best statistics:", best_result.statistics)

    print(f"\nWrote global observable multistart outputs to: {output_path}")
    print(f"Wrote best fit bundle to: {best_fit_output_dir}")

    print("\nWritten summary files:")
    for name, path in written_summary_files.items():
        print(f"  {name}: {path}")

    print("\nWritten best-fit files:")
    for name, path in best_fit_files.items():
        print(f"  {name}: {path}")


def command_multistart_global_observables_variable_projection(
    args: argparse.Namespace,
    config: dict,
) -> None:
    """
    Run global observable multistart fitting using variable projection.

    This is the fast HSQC-style path:

        peak_i(t) = scale_i * observed_species(t) + offset_i

    The nonlinear optimizer fits kinetic parameters only.
    Per-column scale/offset values are solved analytically.
    """

    engine_name = str(config.get("engine_name", "reference"))
    model_path = get_config_value(
        args=args,
        config=config,
        argument_name="model",
        required=True,
    )

    data_path = get_config_value(
        args=args,
        config=config,
        argument_name="data",
        required=True,
    )

    time_column = get_config_value(
        args=args,
        config=config,
        argument_name="time_column",
        default="time",
    )

    signal_columns = get_config_value(
        args=args,
        config=config,
        argument_name="signal_columns",
        default=None,
    )

    exclude_columns = get_config_value(
        args=args,
        config=config,
        argument_name="exclude_columns",
        default=None,
    )

    output_dir = get_config_value(
        args=args,
        config=config,
        argument_name="output_dir",
        required=True,
    )

    observed_species = get_config_value(
        args=args,
        config=config,
        argument_name="observed_species",
        default="A",
    )

    n_starts = int(
        get_config_value(
            args=args,
            config=config,
            argument_name="n_starts",
            default=10,
        )
    )

    random_seed = get_config_value(
        args=args,
        config=config,
        argument_name="random_seed",
        default=None,
    )

    if random_seed is not None:
        random_seed = int(random_seed)

    sort_by = get_config_value(
        args=args,
        config=config,
        argument_name="sort_by",
        default="aic",
    )

    method = get_config_value(
        args=args,
        config=config,
        argument_name="method",
        default="trf",
    )

    loss = get_config_value(
        args=args,
        config=config,
        argument_name="loss",
        default="linear",
    )

    max_nfev = get_config_value(
        args=args,
        config=config,
        argument_name="max_nfev",
        default=None,
    )

    rtol = get_config_value(
        args=args,
        config=config,
        argument_name="rtol",
        default=1e-6,
    )

    atol = get_config_value(
        args=args,
        config=config,
        argument_name="atol",
        default=1e-9,
    )

    fit_scale = bool(config.get("fit_scale", True))
    fit_offset = bool(config.get("fit_offset", True))

    log_uniform = bool(config.get("log_uniform", True))

    if getattr(args, "linear_sampling", False):
        log_uniform = False

    variable_projection_backend = str(
        config.get("variable_projection_backend", "numpy")
    )

    variable_projection_method = str(config.get("variable_projection_method", "LSODA"))

    show_progress = bool(config.get("show_progress", True)) and not bool(
        getattr(args, "no_progress", False)
    )

    model = read_model_file(model_path)

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

    dataset, filtering_result = read_wide_observable_dataset_with_filtering(
        file_path=data_path,
        time_column=time_column,
        signal_columns=signal_columns,
        exclude_columns=exclude_columns,
        max_missing_fraction=peak_filtering_settings["max_missing_fraction"],
        min_initial_intensity=peak_filtering_settings["min_initial_intensity"],
        initial_points=peak_filtering_settings["initial_points"],
        min_dynamic_range=peak_filtering_settings["min_dynamic_range"],
        interpolate_missing=peak_filtering_settings["interpolate_missing"],
    )

    parameter_entries = get_config_list_or_dict_value(
        args=args,
        config=config,
        argument_name="parameter",
        alternative_config_name="parameters",
    )

    parameter_specs = build_parameter_specs(
        model=model,
        parameter_entries=parameter_entries,
        default_guess=float(config.get("default_parameter_guess", 0.1)),
        default_lower=float(config.get("default_parameter_lower", 0.0)),
        default_upper=float(config.get("default_parameter_upper", 100.0)),
    )

    initial_entries = get_config_list_or_dict_value(
        args=args,
        config=config,
        argument_name="initial",
        alternative_config_name="initial_conditions",
    )

    initial_condition_specs = build_initial_condition_specs(
        model=model,
        initial_entries=initial_entries,
    )

    signal_weight_entries = get_config_list_or_dict_value(
        args=args,
        config=config,
        argument_name="signal_weight",
        alternative_config_name="signal_weights",
    )

    settings = FitSettings(
        species_mapping={},
        use_normalized_data=False,
        method=method,
        loss=loss,
        max_nfev=max_nfev,
        rtol=rtol,
        atol=atol,
        signal_weights=parse_signal_weight_entries(signal_weight_entries),
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Running global observable multistart with variable projection")
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Observed species: {observed_species}")
    print(f"Observable columns: {len(dataset.signal_columns)}")
    print(f"Kept observable columns: {len(filtering_result.kept_columns)}")
    print(f"Removed observable columns: {len(filtering_result.removed_columns)}")
    print(f"Starts: {n_starts}")
    print(f"Sort by: {sort_by}")
    print(f"Variable projection backend: {variable_projection_backend}")
    print(f"ODE method: {variable_projection_method}")

    result = fit_global_observable_variable_projection_multistart(
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        observed_species=observed_species,
        settings=settings,
        signal_columns=dataset.signal_columns,
        fit_scale=fit_scale,
        fit_offset=fit_offset,
        backend=variable_projection_backend,
        method=variable_projection_method,
        n_starts=n_starts,
        random_seed=random_seed,
        sort_by=sort_by,
        log_uniform=log_uniform,
        show_progress=show_progress,
        progress_label="Variable projection multistart",
        engine_name=engine_name,
    )

    written_files = export_variable_projection_multistart_summary(
        result=result,
        output_dir=output_path,
        export_best_fit=True,
    )

    peak_filtering_path = write_peak_filtering_table(
        filtering_result=filtering_result,
        output_dir=output_path,
    )

    written_files["peak_filtering"] = peak_filtering_path

    print("\nBest start:", result.best_index)
    print("Best fit success:", result.best_result.success)
    print("Best fit message:", result.best_result.message)
    print("Best fitted kinetic parameters:", result.best_result.fitted_parameters)
    print("Best statistics:", result.best_result.statistics)

    if result.failures:
        print("\nFailed starts:")
        for failure in result.failures:
            print(
                f"  start_{failure.start_index}: "
                f"{failure.error_type}: {failure.error_message}"
            )

    print(f"\nWrote variable projection multistart outputs to: {output_path}")
    print("\nWritten files:")

    for name, path in written_files.items():
        print(f"  {name}: {path}")


def command_compare_global_observables(args: argparse.Namespace) -> None:
    """
    Compare several global observable mechanisms on one dataset.

    Intended first use case:
        compare mechanisms on assigned HSQC peak intensity data

    Example models:
        A>B
        A-B
        2A>A2
        2A<->A2
    """

    config = load_fit_config(args.config)

    use_multispecies_variable_projection = bool(
        config.get("use_multispecies_variable_projection", False)
    )

    if use_multispecies_variable_projection:
        command_compare_global_observables_multispecies_variable_projection(
            args=args,
            config=config,
        )
        return

    use_variable_projection = bool(
        config.get("use_variable_projection", False)
    ) or bool(getattr(args, "variable_projection", False))

    if use_variable_projection:
        command_compare_global_observables_variable_projection(
            args=args,
            config=config,
        )
        return
    data_path = get_config_value(
        args=args,
        config=config,
        argument_name="data",
        required=True,
    )

    time_column = get_config_value(
        args=args,
        config=config,
        argument_name="time_column",
        default="time",
    )

    signal_columns = get_config_value(
        args=args,
        config=config,
        argument_name="signal_columns",
        default=None,
    )

    exclude_columns = get_config_value(
        args=args,
        config=config,
        argument_name="exclude_columns",
        default=None,
    )

    output_dir = get_config_value(
        args=args,
        config=config,
        argument_name="output_dir",
        required=True,
    )

    observed_species = get_config_value(
        args=args,
        config=config,
        argument_name="observed_species",
        default="A",
    )

    observed_species_by_model = config.get(
        "observed_species_by_model",
        observed_species,
    )

    sort_by = get_config_value(
        args=args,
        config=config,
        argument_name="sort_by",
        default="aic",
    )

    default_parameter_guess = get_config_value(
        args=args,
        config=config,
        argument_name="default_parameter_guess",
        default=0.1,
    )

    default_parameter_lower = get_config_value(
        args=args,
        config=config,
        argument_name="default_parameter_lower",
        default=0.0,
    )

    default_parameter_upper = get_config_value(
        args=args,
        config=config,
        argument_name="default_parameter_upper",
        default=100.0,
    )

    method = get_config_value(
        args=args,
        config=config,
        argument_name="method",
        default="trf",
    )

    loss = get_config_value(
        args=args,
        config=config,
        argument_name="loss",
        default="linear",
    )

    max_nfev = get_config_value(
        args=args,
        config=config,
        argument_name="max_nfev",
        default=None,
    )

    rtol = get_config_value(
        args=args,
        config=config,
        argument_name="rtol",
        default=1e-6,
    )

    atol = get_config_value(
        args=args,
        config=config,
        argument_name="atol",
        default=1e-9,
    )

    fit_scale = bool(config.get("fit_scale", True))
    fit_offset = bool(config.get("fit_offset", True))

    scale_initial_guess = float(config.get("scale_initial_guess", 1.0))
    scale_lower_bound = float(config.get("scale_lower_bound", 0.0))
    scale_upper_bound = float(config.get("scale_upper_bound", float("inf")))

    offset_initial_guess = float(config.get("offset_initial_guess", 0.0))
    offset_lower_bound = float(config.get("offset_lower_bound", -float("inf")))
    offset_upper_bound = float(config.get("offset_upper_bound", float("inf")))

    no_plots = bool(config.get("no_plots", False)) or bool(args.no_plots)

    models = build_model_specs_from_comparison_config(config)

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

    dataset, filtering_result = read_wide_observable_dataset_with_filtering(
        file_path=data_path,
        time_column=time_column,
        signal_columns=signal_columns,
        exclude_columns=exclude_columns,
        max_missing_fraction=peak_filtering_settings["max_missing_fraction"],
        min_initial_intensity=peak_filtering_settings["min_initial_intensity"],
        initial_points=peak_filtering_settings["initial_points"],
        min_dynamic_range=peak_filtering_settings["min_dynamic_range"],
        interpolate_missing=peak_filtering_settings["interpolate_missing"],
    )

    parameter_specs_by_model = build_parameter_specs_by_model_from_config(
        models=models,
        config=config,
        default_guess=default_parameter_guess,
        default_lower=default_parameter_lower,
        default_upper=default_parameter_upper,
    )

    initial_condition_specs_by_model = (
        build_initial_condition_specs_by_model_from_config(
            models=models,
            config=config,
        )
    )

    signal_weight_entries = get_config_list_or_dict_value(
        args=args,
        config=config,
        argument_name="signal_weight",
        alternative_config_name="signal_weights",
    )

    settings = FitSettings(
        species_mapping={},
        use_normalized_data=False,
        method=method,
        loss=loss,
        max_nfev=max_nfev,
        rtol=rtol,
        atol=atol,
        signal_weights=parse_signal_weight_entries(signal_weight_entries),
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Comparing {len(models)} global observable models")
    print("Models:", ", ".join(models))
    print("Observable columns:", len(dataset.signal_columns))
    print("Kept observable columns:", len(filtering_result.kept_columns))
    print("Removed observable columns:", len(filtering_result.removed_columns))

    comparison_result = fit_global_observable_model_comparison(
        models=models,
        dataset=dataset,
        parameter_specs_by_model=parameter_specs_by_model,
        initial_condition_specs_by_model=initial_condition_specs_by_model,
        observed_species_by_model=observed_species_by_model,
        settings_by_model=settings,
        signal_columns=dataset.signal_columns,
        fit_scale=fit_scale,
        fit_offset=fit_offset,
        scale_initial_guess=scale_initial_guess,
        scale_lower_bound=scale_lower_bound,
        scale_upper_bound=scale_upper_bound,
        offset_initial_guess=offset_initial_guess,
        offset_lower_bound=offset_lower_bound,
        offset_upper_bound=offset_upper_bound,
        sort_by=sort_by,
    )

    written_summary_files = export_global_observable_model_comparison(
        result=comparison_result,
        output_dir=output_path,
    )

    peak_filtering_path = write_peak_filtering_table(
        filtering_result=filtering_result,
        output_dir=output_path,
    )

    written_summary_files["peak_filtering"] = peak_filtering_path

    best_model_name = comparison_result.best_model_name
    best_fit_output = comparison_result.fit_outputs[best_model_name]
    best_fit_result = comparison_result.best_fit_result
    best_model = models[best_model_name]

    best_fit_output_dir = output_path / "best_fit"

    best_fit_files = export_fit_bundle(
        fit_result=best_fit_result,
        model=best_model,
        dataset=dataset,
        output_dir=best_fit_output_dir,
        parameter_specs=parameter_specs_by_model[best_model_name],
        initial_condition_specs=initial_condition_specs_by_model[best_model_name],
        observable_specs=best_fit_output.observable_specs,
        species_mapping={},
        include_plots=not no_plots,
        fit_settings=settings,
        command="compare-global-observables-best-fit",
        config_path=args.config,
        extra_run_metadata={
            "best_model_name": best_model_name,
            "compared_models": list(models),
            "sort_by": sort_by,
            "n_observable_columns": len(dataset.signal_columns),
        },
    )

    print("\nBest model:", best_model_name)
    print("Best fit success:", best_fit_result.success)
    print("Best fit message:", best_fit_result.message)
    print("Best fitted kinetic parameters:", best_fit_result.fitted_parameters)
    print("Best statistics:", best_fit_result.statistics)

    if comparison_result.failures:
        print("\nFailed models:")
        for failure in comparison_result.failures:
            print(
                f"  {failure.model_name}: {failure.error_type}: {failure.error_message}"
            )

    print(f"\nWrote model comparison outputs to: {output_path}")
    print(f"Wrote best fit bundle to: {best_fit_output_dir}")

    print("\nWritten summary files:")
    for name, path in written_summary_files.items():
        print(f"  {name}: {path}")

    print("\nWritten best-fit files:")
    for name, path in best_fit_files.items():
        print(f"  {name}: {path}")


def command_compare_global_observables_variable_projection(
    args: argparse.Namespace,
    config: dict,
) -> None:
    """
    Compare global observable mechanisms using variable projection.

    This is the fast HSQC-style model-comparison path:

        peak_i(t) = scale_i * observed_species(t) + offset_i

    Each candidate mechanism is fit using nonlinear kinetic parameters only.
    Per-column scale/offset values are solved analytically.
    """

    engine_name = str(config.get("engine_name", "reference"))
    data_path = get_config_value(
        args=args,
        config=config,
        argument_name="data",
        required=True,
    )

    time_column = get_config_value(
        args=args,
        config=config,
        argument_name="time_column",
        default="time",
    )

    signal_columns = get_config_value(
        args=args,
        config=config,
        argument_name="signal_columns",
        default=None,
    )

    exclude_columns = get_config_value(
        args=args,
        config=config,
        argument_name="exclude_columns",
        default=None,
    )

    output_dir = get_config_value(
        args=args,
        config=config,
        argument_name="output_dir",
        required=True,
    )

    observed_species = get_config_value(
        args=args,
        config=config,
        argument_name="observed_species",
        default="A",
    )

    observed_species_by_model = config.get(
        "observed_species_by_model",
        observed_species,
    )

    sort_by = get_config_value(
        args=args,
        config=config,
        argument_name="sort_by",
        default="aic",
    )

    method = get_config_value(
        args=args,
        config=config,
        argument_name="method",
        default="trf",
    )

    loss = get_config_value(
        args=args,
        config=config,
        argument_name="loss",
        default="linear",
    )

    max_nfev = get_config_value(
        args=args,
        config=config,
        argument_name="max_nfev",
        default=None,
    )

    rtol = get_config_value(
        args=args,
        config=config,
        argument_name="rtol",
        default=1e-6,
    )

    atol = get_config_value(
        args=args,
        config=config,
        argument_name="atol",
        default=1e-9,
    )

    fit_scale = bool(config.get("fit_scale", True))
    fit_offset = bool(config.get("fit_offset", True))

    variable_projection_backend = str(
        config.get("variable_projection_backend", "numpy")
    )

    variable_projection_method = str(config.get("variable_projection_method", "LSODA"))

    models = build_model_specs_from_comparison_config(config)

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

    dataset, filtering_result = read_wide_observable_dataset_with_filtering(
        file_path=data_path,
        time_column=time_column,
        signal_columns=signal_columns,
        exclude_columns=exclude_columns,
        max_missing_fraction=peak_filtering_settings["max_missing_fraction"],
        min_initial_intensity=peak_filtering_settings["min_initial_intensity"],
        initial_points=peak_filtering_settings["initial_points"],
        min_dynamic_range=peak_filtering_settings["min_dynamic_range"],
        interpolate_missing=peak_filtering_settings["interpolate_missing"],
    )

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

    signal_weight_entries = get_config_list_or_dict_value(
        args=args,
        config=config,
        argument_name="signal_weight",
        alternative_config_name="signal_weights",
    )

    settings = FitSettings(
        species_mapping={},
        use_normalized_data=False,
        method=method,
        loss=loss,
        max_nfev=max_nfev,
        rtol=rtol,
        atol=atol,
        signal_weights=parse_signal_weight_entries(signal_weight_entries),
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Running global observable model comparison with variable projection")
    print("Models:", ", ".join(models))
    print(f"Data: {data_path}")
    print(f"Observed species: {observed_species}")
    print(f"Observable columns: {len(dataset.signal_columns)}")
    print(f"Kept observable columns: {len(filtering_result.kept_columns)}")
    print(f"Removed observable columns: {len(filtering_result.removed_columns)}")
    print(f"Sort by: {sort_by}")
    print(f"Variable projection backend: {variable_projection_backend}")
    print(f"ODE method: {variable_projection_method}")

    result = fit_global_observable_variable_projection_model_comparison(
        models=models,
        dataset=dataset,
        parameter_specs_by_model=parameter_specs_by_model,
        initial_condition_specs_by_model=initial_condition_specs_by_model,
        observed_species_by_model=observed_species_by_model,
        settings_by_model=settings,
        signal_columns=dataset.signal_columns,
        fit_scale=fit_scale,
        fit_offset=fit_offset,
        backend=variable_projection_backend,
        method=variable_projection_method,
        sort_by=sort_by,
        engine_name=engine_name,
    )

    written_files = export_variable_projection_model_comparison(
        result=result,
        output_dir=output_path,
        export_best_fit=True,
    )

    peak_filtering_path = write_peak_filtering_table(
        filtering_result=filtering_result,
        output_dir=output_path,
    )

    written_files["peak_filtering"] = peak_filtering_path

    print("\nBest model:", result.best_model_name)
    print("Best fit success:", result.best_result.success)
    print("Best fit message:", result.best_result.message)
    print("Best fitted kinetic parameters:", result.best_result.fitted_parameters)
    print("Best statistics:", result.best_result.statistics)

    if result.failures:
        print("\nFailed models:")
        for failure in result.failures:
            print(
                f"  {failure.model_name}: {failure.error_type}: {failure.error_message}"
            )

    print(f"\nWrote variable projection model comparison outputs to: {output_path}")
    print("\nWritten files:")

    for name, path in written_files.items():
        print(f"  {name}: {path}")


def command_multistart_compare_global_observables(args: argparse.Namespace) -> None:
    """
    Compare several global observable mechanisms using multistart per model.

    Intended use case:
        robust model comparison for assigned HSQC peak intensity data

    Workflow:
        for each candidate model:
            run global observable multistart
            keep the model's best fit

        then:
            rank each model's best fit by AIC/BIC/RMSE/RSS/etc.
    """

    config = load_fit_config(args.config)

    use_multispecies_variable_projection = bool(
        config.get("use_multispecies_variable_projection", False)
    )

    if use_multispecies_variable_projection:
        command_multistart_compare_global_observables_multispecies_variable_projection(
            args=args,
            config=config,
        )
        return

    use_variable_projection = bool(
        config.get("use_variable_projection", False)
    ) or bool(getattr(args, "variable_projection", False))

    if use_variable_projection:
        command_multistart_compare_global_observables_variable_projection(
            args=args,
            config=config,
        )
        return

    data_path = get_config_value(
        args=args,
        config=config,
        argument_name="data",
        required=True,
    )

    time_column = get_config_value(
        args=args,
        config=config,
        argument_name="time_column",
        default="time",
    )

    signal_columns = get_config_value(
        args=args,
        config=config,
        argument_name="signal_columns",
        default=None,
    )

    exclude_columns = get_config_value(
        args=args,
        config=config,
        argument_name="exclude_columns",
        default=None,
    )

    output_dir = get_config_value(
        args=args,
        config=config,
        argument_name="output_dir",
        required=True,
    )

    observed_species = get_config_value(
        args=args,
        config=config,
        argument_name="observed_species",
        default="A",
    )

    observed_species_by_model = config.get(
        "observed_species_by_model",
        observed_species,
    )

    sort_by = get_config_value(
        args=args,
        config=config,
        argument_name="sort_by",
        default="aic",
    )

    multistart_sort_by = config.get("multistart_sort_by", sort_by)

    n_starts = int(
        get_config_value(
            args=args,
            config=config,
            argument_name="n_starts",
            default=10,
        )
    )

    n_workers = get_config_value(
        args=args,
        config=config,
        argument_name="n_workers",
        default=1,
    )

    if n_workers is not None:
        n_workers = int(n_workers)

    random_seed = get_config_value(
        args=args,
        config=config,
        argument_name="random_seed",
        default=None,
    )

    if random_seed is not None:
        random_seed = int(random_seed)

    default_parameter_guess = get_config_value(
        args=args,
        config=config,
        argument_name="default_parameter_guess",
        default=0.1,
    )

    default_parameter_lower = get_config_value(
        args=args,
        config=config,
        argument_name="default_parameter_lower",
        default=0.0,
    )

    default_parameter_upper = get_config_value(
        args=args,
        config=config,
        argument_name="default_parameter_upper",
        default=100.0,
    )

    method = get_config_value(
        args=args,
        config=config,
        argument_name="method",
        default="trf",
    )

    loss = get_config_value(
        args=args,
        config=config,
        argument_name="loss",
        default="linear",
    )

    max_nfev = get_config_value(
        args=args,
        config=config,
        argument_name="max_nfev",
        default=None,
    )

    rtol = get_config_value(
        args=args,
        config=config,
        argument_name="rtol",
        default=1e-6,
    )

    atol = get_config_value(
        args=args,
        config=config,
        argument_name="atol",
        default=1e-9,
    )

    fit_scale = bool(config.get("fit_scale", True))
    fit_offset = bool(config.get("fit_offset", True))

    scale_initial_guess = float(config.get("scale_initial_guess", 1.0))
    scale_lower_bound = float(config.get("scale_lower_bound", 0.0))
    scale_upper_bound = float(config.get("scale_upper_bound", float("inf")))

    offset_initial_guess = float(config.get("offset_initial_guess", 0.0))
    offset_lower_bound = float(config.get("offset_lower_bound", -float("inf")))
    offset_upper_bound = float(config.get("offset_upper_bound", float("inf")))

    log_uniform_parameters = bool(config.get("log_uniform", True))

    if args.linear_sampling:
        log_uniform_parameters = False

    randomize_observable_scales = bool(config.get("randomize_observable_scales", True))

    randomize_observable_offsets = bool(
        config.get("randomize_observable_offsets", True)
    )

    log_uniform_observable_scales = bool(
        config.get("log_uniform_observable_scales", False)
    )

    no_plots = bool(config.get("no_plots", False)) or bool(args.no_plots)

    models = build_model_specs_from_comparison_config(config)

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

    dataset, filtering_result = read_wide_observable_dataset_with_filtering(
        file_path=data_path,
        time_column=time_column,
        signal_columns=signal_columns,
        exclude_columns=exclude_columns,
        max_missing_fraction=peak_filtering_settings["max_missing_fraction"],
        min_initial_intensity=peak_filtering_settings["min_initial_intensity"],
        initial_points=peak_filtering_settings["initial_points"],
        min_dynamic_range=peak_filtering_settings["min_dynamic_range"],
        interpolate_missing=peak_filtering_settings["interpolate_missing"],
    )

    parameter_specs_by_model = build_parameter_specs_by_model_from_config(
        models=models,
        config=config,
        default_guess=default_parameter_guess,
        default_lower=default_parameter_lower,
        default_upper=default_parameter_upper,
    )

    initial_condition_specs_by_model = (
        build_initial_condition_specs_by_model_from_config(
            models=models,
            config=config,
        )
    )

    signal_weight_entries = get_config_list_or_dict_value(
        args=args,
        config=config,
        argument_name="signal_weight",
        alternative_config_name="signal_weights",
    )

    settings = FitSettings(
        species_mapping={},
        use_normalized_data=False,
        method=method,
        loss=loss,
        max_nfev=max_nfev,
        rtol=rtol,
        atol=atol,
        signal_weights=parse_signal_weight_entries(signal_weight_entries),
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Running multistart comparison for {len(models)} models")
    print("Models:", ", ".join(models))
    print("Starts per model:", n_starts)
    print("Workers:", n_workers)
    print("Observable columns:", len(dataset.signal_columns))
    print("Kept observable columns:", len(filtering_result.kept_columns))
    print("Removed observable columns:", len(filtering_result.removed_columns))

    comparison_result = fit_global_observable_multistart_model_comparison(
        models=models,
        dataset=dataset,
        parameter_specs_by_model=parameter_specs_by_model,
        initial_condition_specs_by_model=initial_condition_specs_by_model,
        observed_species_by_model=observed_species_by_model,
        settings_by_model=settings,
        signal_columns=dataset.signal_columns,
        fit_scale=fit_scale,
        fit_offset=fit_offset,
        scale_initial_guess=scale_initial_guess,
        scale_lower_bound=scale_lower_bound,
        scale_upper_bound=scale_upper_bound,
        offset_initial_guess=offset_initial_guess,
        offset_lower_bound=offset_lower_bound,
        offset_upper_bound=offset_upper_bound,
        n_starts=n_starts,
        n_workers=n_workers,
        random_seed=random_seed,
        sort_by=sort_by,
        multistart_sort_by=multistart_sort_by,
        log_uniform_parameters=log_uniform_parameters,
        randomize_observable_scales=randomize_observable_scales,
        randomize_observable_offsets=randomize_observable_offsets,
        log_uniform_observable_scales=log_uniform_observable_scales,
    )

    written_summary_files = export_global_observable_multistart_model_comparison(
        result=comparison_result,
        output_dir=output_path,
        export_per_model_summaries=True,
    )

    peak_filtering_path = write_peak_filtering_table(
        filtering_result=filtering_result,
        output_dir=output_path,
    )

    written_summary_files["peak_filtering"] = peak_filtering_path

    best_model_name = comparison_result.best_model_name
    best_fit_result = comparison_result.best_fit_result
    best_model = models[best_model_name]

    if isinstance(observed_species_by_model, dict):
        best_observed_species = observed_species_by_model.get(
            best_model_name,
            "A",
        )
    else:
        best_observed_species = observed_species_by_model

    best_observable_specs = build_shared_species_observable_specs(
        signal_columns=dataset.signal_columns,
        species=best_observed_species,
        fit_scale=fit_scale,
        fit_offset=fit_offset,
        scale_initial_guess=scale_initial_guess,
        scale_lower_bound=scale_lower_bound,
        scale_upper_bound=scale_upper_bound,
        offset_initial_guess=offset_initial_guess,
        offset_lower_bound=offset_lower_bound,
        offset_upper_bound=offset_upper_bound,
    )

    best_fit_output_dir = output_path / "best_fit"

    best_fit_files = export_fit_bundle(
        fit_result=best_fit_result,
        model=best_model,
        dataset=dataset,
        output_dir=best_fit_output_dir,
        parameter_specs=parameter_specs_by_model[best_model_name],
        initial_condition_specs=initial_condition_specs_by_model[best_model_name],
        observable_specs=best_observable_specs,
        species_mapping={},
        include_plots=not no_plots,
        fit_settings=settings,
        command="multistart-compare-global-observables-best-fit",
        config_path=args.config,
        extra_run_metadata={
            "best_model_name": best_model_name,
            "compared_models": list(models),
            "sort_by": sort_by,
            "multistart_sort_by": multistart_sort_by,
            "n_starts": n_starts,
            "n_workers": n_workers,
            "n_observable_columns": len(dataset.signal_columns),
        },
    )

    print("\nBest model:", best_model_name)
    print("Best fit success:", best_fit_result.success)
    print("Best fit message:", best_fit_result.message)
    print("Best fitted kinetic parameters:", best_fit_result.fitted_parameters)
    print("Best statistics:", best_fit_result.statistics)

    if comparison_result.failures:
        print("\nFailed models:")
        for failure in comparison_result.failures:
            print(
                f"  {failure.model_name}: {failure.error_type}: {failure.error_message}"
            )

    print(f"\nWrote multistart model comparison outputs to: {output_path}")
    print(f"Wrote best fit bundle to: {best_fit_output_dir}")

    print("\nWritten summary files:")
    for name, path in written_summary_files.items():
        print(f"  {name}: {path}")

    print("\nWritten best-fit files:")
    for name, path in best_fit_files.items():
        print(f"  {name}: {path}")


def command_multistart_compare_global_observables_variable_projection(
    args: argparse.Namespace,
    config: dict,
) -> None:
    """
    Compare several global observable mechanisms using variable projection
    multistart fitting.

    This is the fast robust HSQC path:
        for each model:
            run variable-projection multistart
            keep best start

        then:
            rank models by AIC/BIC/RMSE/RSS.
    """

    engine_name = str(config.get("engine_name", "reference"))
    data_path = get_config_value(
        args=args,
        config=config,
        argument_name="data",
        required=True,
    )

    time_column = get_config_value(
        args=args,
        config=config,
        argument_name="time_column",
        default="time",
    )

    signal_columns = get_config_value(
        args=args,
        config=config,
        argument_name="signal_columns",
        default=None,
    )

    exclude_columns = get_config_value(
        args=args,
        config=config,
        argument_name="exclude_columns",
        default=None,
    )

    output_dir = get_config_value(
        args=args,
        config=config,
        argument_name="output_dir",
        required=True,
    )

    observed_species = get_config_value(
        args=args,
        config=config,
        argument_name="observed_species",
        default="A",
    )

    observed_species_by_model = config.get(
        "observed_species_by_model",
        observed_species,
    )

    sort_by = get_config_value(
        args=args,
        config=config,
        argument_name="sort_by",
        default="bic",
    )

    multistart_sort_by = config.get("multistart_sort_by", sort_by)

    n_starts = int(
        get_config_value(
            args=args,
            config=config,
            argument_name="n_starts",
            default=10,
        )
    )

    random_seed = get_config_value(
        args=args,
        config=config,
        argument_name="random_seed",
        default=None,
    )

    if random_seed is not None:
        random_seed = int(random_seed)

    default_parameter_guess = get_config_value(
        args=args,
        config=config,
        argument_name="default_parameter_guess",
        default=0.1,
    )

    default_parameter_lower = get_config_value(
        args=args,
        config=config,
        argument_name="default_parameter_lower",
        default=0.0,
    )

    default_parameter_upper = get_config_value(
        args=args,
        config=config,
        argument_name="default_parameter_upper",
        default=100.0,
    )

    method = get_config_value(
        args=args,
        config=config,
        argument_name="method",
        default="trf",
    )

    loss = get_config_value(
        args=args,
        config=config,
        argument_name="loss",
        default="linear",
    )

    max_nfev = get_config_value(
        args=args,
        config=config,
        argument_name="max_nfev",
        default=None,
    )

    rtol = get_config_value(
        args=args,
        config=config,
        argument_name="rtol",
        default=1e-6,
    )

    atol = get_config_value(
        args=args,
        config=config,
        argument_name="atol",
        default=1e-9,
    )

    fit_scale = bool(config.get("fit_scale", True))
    fit_offset = bool(config.get("fit_offset", True))

    log_uniform = bool(config.get("log_uniform", True))

    if getattr(args, "linear_sampling", False):
        log_uniform = False

    variable_projection_backend = str(
        config.get("variable_projection_backend", "numpy")
    )

    variable_projection_method = str(config.get("variable_projection_method", "LSODA"))

    show_progress = bool(config.get("show_progress", True)) and not bool(
        getattr(args, "no_progress", False)
    )

    no_plots = bool(config.get("no_plots", False)) or bool(args.no_plots)

    models = build_model_specs_from_comparison_config(config)

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

    dataset, filtering_result = read_wide_observable_dataset_with_filtering(
        file_path=data_path,
        time_column=time_column,
        signal_columns=signal_columns,
        exclude_columns=exclude_columns,
        max_missing_fraction=peak_filtering_settings["max_missing_fraction"],
        min_initial_intensity=peak_filtering_settings["min_initial_intensity"],
        initial_points=peak_filtering_settings["initial_points"],
        min_dynamic_range=peak_filtering_settings["min_dynamic_range"],
        interpolate_missing=peak_filtering_settings["interpolate_missing"],
    )

    parameter_specs_by_model = build_parameter_specs_by_model_from_config(
        models=models,
        config=config,
        default_guess=default_parameter_guess,
        default_lower=default_parameter_lower,
        default_upper=default_parameter_upper,
    )

    initial_condition_specs_by_model = (
        build_initial_condition_specs_by_model_from_config(
            models=models,
            config=config,
        )
    )

    signal_weight_entries = get_config_list_or_dict_value(
        args=args,
        config=config,
        argument_name="signal_weight",
        alternative_config_name="signal_weights",
    )

    settings = FitSettings(
        species_mapping={},
        use_normalized_data=False,
        method=method,
        loss=loss,
        max_nfev=max_nfev,
        rtol=rtol,
        atol=atol,
        signal_weights=parse_signal_weight_entries(signal_weight_entries),
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Running variable projection multistart model comparison")
    print("Models:", ", ".join(models))
    print("Observable columns:", len(dataset.signal_columns))
    print("Kept observable columns:", len(filtering_result.kept_columns))
    print("Removed observable columns:", len(filtering_result.removed_columns))
    print("Starts per model:", n_starts)
    print("Model ranking sort by:", sort_by)
    print("Within-model multistart sort by:", multistart_sort_by)
    print("Variable projection backend:", variable_projection_backend)
    print("ODE method:", variable_projection_method)

    comparison_result = (
        fit_global_observable_variable_projection_multistart_model_comparison(
            models=models,
            dataset=dataset,
            parameter_specs_by_model=parameter_specs_by_model,
            initial_condition_specs_by_model=initial_condition_specs_by_model,
            observed_species_by_model=observed_species_by_model,
            settings=settings,
            signal_columns=dataset.signal_columns,
            fit_scale=fit_scale,
            fit_offset=fit_offset,
            backend=variable_projection_backend,
            method=variable_projection_method,
            n_starts=n_starts,
            random_seed=random_seed,
            sort_by=sort_by,
            multistart_sort_by=multistart_sort_by,
            log_uniform=log_uniform,
            show_progress=show_progress,
            engine_name=engine_name,
        )
    )

    written_summary_files = export_variable_projection_multistart_model_comparison(
        result=comparison_result,
        output_dir=output_path,
    )

    peak_filtering_path = write_peak_filtering_table(
        filtering_result=filtering_result,
        output_dir=output_path,
    )

    written_summary_files["peak_filtering"] = peak_filtering_path

    best_fit_result = comparison_result.best_fit_result

    print("\nBest model:", comparison_result.best_model_name)
    print("Best fit success:", best_fit_result.success)
    print("Best fit message:", best_fit_result.message)
    print(
        "Best fitted kinetic parameters:",
        best_fit_result.fitted_parameters,
    )
    print("Best statistics:", best_fit_result.statistics)

    if comparison_result.failures:
        print("\nFailed models:")
        for failure in comparison_result.failures:
            print(
                f"  {failure.model_name}: {failure.error_type}: {failure.error_message}"
            )

    print(
        f"\nWrote variable projection multistart model comparison outputs to: {output_path}"
    )

    print("\nWritten files:")
    for name, path in written_summary_files.items():
        print(f"  {name}: {path}")


def command_profile_likelihood_global_observables(args: argparse.Namespace) -> None:
    config = load_fit_config(args.config)
    engine_name = str(config.get("engine_name", "reference"))

    if bool(config.get("use_multispecies_variable_projection", False)):
        command_profile_likelihood_global_observables_multispecies_variable_projection(
            args=args,
            config=config,
        )
        return

    model_path = config["model"]
    data_path = config["data"]
    time_column = config.get("time_column", "time")
    signal_columns = config.get("signal_columns")
    exclude_columns = config.get("exclude_columns")
    observed_species = config.get("observed_species", "A")
    output_dir = config["output_dir"]

    model = read_model_file(model_path)

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

    dataset, filtering_result = read_wide_observable_dataset_with_filtering(
        file_path=data_path,
        time_column=time_column,
        signal_columns=signal_columns,
        exclude_columns=exclude_columns,
        max_missing_fraction=peak_filtering_settings["max_missing_fraction"],
        min_initial_intensity=peak_filtering_settings["min_initial_intensity"],
        initial_points=peak_filtering_settings["initial_points"],
        min_dynamic_range=peak_filtering_settings["min_dynamic_range"],
        interpolate_missing=peak_filtering_settings["interpolate_missing"],
    )

    parameter_entries = config.get("parameter")
    if parameter_entries is None:
        parameter_entries = config.get("parameters")

    initial_entries = config.get("initial")
    if initial_entries is None:
        initial_entries = config.get("initial_conditions")

    signal_weight_entries = config.get("signal_weight")
    if signal_weight_entries is None:
        signal_weight_entries = config.get("signal_weights")

    parameter_specs = build_parameter_specs(
        model=model,
        parameter_entries=parameter_entries,
        default_guess=float(config.get("default_parameter_guess", 0.1)),
        default_lower=float(config.get("default_parameter_lower", 0.0)),
        default_upper=float(config.get("default_parameter_upper", 100.0)),
    )

    initial_condition_specs = build_initial_condition_specs(
        model=model,
        initial_entries=initial_entries,
    )

    settings = FitSettings(
        species_mapping={},
        use_normalized_data=False,
        method=config.get("method", "trf"),
        loss=config.get("loss", "linear"),
        max_nfev=config.get("max_nfev"),
        rtol=config.get("rtol", 1e-6),
        atol=config.get("atol", 1e-9),
        signal_weights=parse_signal_weight_entries(signal_weight_entries),
    )

    result = fit_variable_projection_profile_likelihood(
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        observed_species=observed_species,
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
        show_progress=not bool(getattr(args, "no_progress", False)),
        engine_name=engine_name,
    )

    output_path = Path(output_dir)
    written_files = export_profile_likelihood_result(
        result=result,
        output_dir=output_path,
    )

    peak_filtering_path = write_peak_filtering_table(
        filtering_result=filtering_result,
        output_dir=output_path,
    )

    written_files["peak_filtering"] = peak_filtering_path

    print("\nProfile likelihood complete")
    print(result.profile_table.to_string(index=False))

    identifiability_path = output_path / "identifiability_warnings.csv"

    if identifiability_path.exists() and identifiability_path.stat().st_size > 0:
        try:
            warnings_table = pd.read_csv(identifiability_path)
        except pd.errors.EmptyDataError:
            warnings_table = pd.DataFrame()

        if not warnings_table.empty:
            print("\nIdentifiability warnings:")
            print(warnings_table.to_string(index=False))

    print(f"\nWrote outputs to: {output_path}")

    for name, path in written_files.items():
        print(f"  {name}: {path}")


def command_bootstrap_global_observables(args: argparse.Namespace) -> None:
    """
    Bootstrap uncertainty estimates for global observable fitting.

    Currently supports the fast variable-projection HSQC path.
    """

    config = load_fit_config(args.config)
    engine_name = str(config.get("engine_name", "reference"))

    if bool(config.get("use_multispecies_variable_projection", False)):
        command_bootstrap_global_observables_multispecies_variable_projection(
            args=args,
            config=config,
        )
        return

    use_multispecies_variable_projection = bool(
        config.get("use_multispecies_variable_projection", False)
    )

    if use_multispecies_variable_projection:
        command_compare_global_observables_multispecies_variable_projection(
            args=args,
            config=config,
        )
        return

    use_variable_projection = bool(
        config.get("use_variable_projection", False)
    ) or bool(getattr(args, "variable_projection", False))

    if not use_variable_projection:
        raise ValueError(
            "bootstrap-global-observables currently requires "
            "--variable-projection or use_variable_projection=true."
        )

    model_path = config["model"]
    data_path = config["data"]

    time_column = config.get("time_column", "time")
    signal_columns = config.get("signal_columns")
    exclude_columns = config.get("exclude_columns")
    observed_species = config.get("observed_species", "A")
    output_dir = config["output_dir"]
    n_bootstrap = int(config.get("n_bootstrap", 100))
    n_workers = int(config.get("n_workers", 1))

    random_seed = config.get("random_seed")

    if random_seed is not None:
        random_seed = int(random_seed)

    confidence_level = float(config.get("confidence_level", 0.95))

    method = config.get("method", "trf")
    loss = config.get("loss", "linear")
    max_nfev = config.get("max_nfev")
    rtol = config.get("rtol", 1e-6)
    atol = config.get("atol", 1e-9)

    fit_scale = bool(config.get("fit_scale", True))
    fit_offset = bool(config.get("fit_offset", True))

    variable_projection_backend = str(
        config.get("variable_projection_backend", "numpy")
    )

    variable_projection_method = str(config.get("variable_projection_method", "LSODA"))

    show_progress = bool(config.get("show_progress", True)) and not bool(
        getattr(args, "no_progress", False)
    )

    model = read_model_file(model_path)
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

    dataset, filtering_result = read_wide_observable_dataset_with_filtering(
        file_path=data_path,
        time_column=time_column,
        signal_columns=signal_columns,
        exclude_columns=exclude_columns,
        max_missing_fraction=peak_filtering_settings["max_missing_fraction"],
        min_initial_intensity=peak_filtering_settings["min_initial_intensity"],
        initial_points=peak_filtering_settings["initial_points"],
        min_dynamic_range=peak_filtering_settings["min_dynamic_range"],
        interpolate_missing=peak_filtering_settings["interpolate_missing"],
    )

    parameter_entries = config.get("parameter")

    if parameter_entries is None:
        parameter_entries = config.get("parameters")

    parameter_specs = build_parameter_specs(
        model=model,
        parameter_entries=parameter_entries,
        default_guess=float(config.get("default_parameter_guess", 0.1)),
        default_lower=float(config.get("default_parameter_lower", 0.0)),
        default_upper=float(config.get("default_parameter_upper", 100.0)),
    )

    initial_entries = config.get("initial")

    if initial_entries is None:
        initial_entries = config.get("initial_conditions")

    initial_condition_specs = build_initial_condition_specs(
        model=model,
        initial_entries=initial_entries,
    )

    signal_weight_entries = config.get("signal_weight")

    if signal_weight_entries is None:
        signal_weight_entries = config.get("signal_weights")

    settings = FitSettings(
        species_mapping={},
        use_normalized_data=False,
        method=method,
        loss=loss,
        max_nfev=max_nfev,
        rtol=rtol,
        atol=atol,
        signal_weights=parse_signal_weight_entries(signal_weight_entries),
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Running variable projection bootstrap")
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Observed species: {observed_species}")
    print(f"Observable columns: {len(dataset.signal_columns)}")
    print(f"Bootstrap replicates: {n_bootstrap}")
    print(f"Workers: {n_workers}")
    print(f"Confidence level: {confidence_level}")
    print(f"Backend: {variable_projection_backend}")
    print(f"ODE method: {variable_projection_method}")

    result = bootstrap_global_observable_variable_projection_fit(
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        observed_species=observed_species,
        settings=settings,
        signal_columns=dataset.signal_columns,
        fit_scale=fit_scale,
        fit_offset=fit_offset,
        backend=variable_projection_backend,
        method=variable_projection_method,
        n_bootstrap=n_bootstrap,
        n_workers=n_workers,
        random_seed=random_seed,
        confidence_level=confidence_level,
        show_progress=show_progress,
        engine_name=engine_name,
    )

    written_files = export_variable_projection_bootstrap_result(
        result=result,
        output_dir=output_path,
        export_original_fit=True,
        include_plots=not bool(config.get("no_plots", False)),
    )

    peak_filtering_path = write_peak_filtering_table(
        filtering_result=filtering_result,
        output_dir=output_path,
    )

    written_files["peak_filtering"] = peak_filtering_path

    print("\nBootstrap successful fits:", len(result.bootstrap_results))
    print("Bootstrap failed fits:", len(result.failures))
    print("\nParameter uncertainty summary:")
    print(result.summary_table.to_string(index=False))

    identifiability_path = output_path / "identifiability_warnings.csv"

    if identifiability_path.exists() and identifiability_path.stat().st_size > 0:
        try:
            warnings_table = pd.read_csv(identifiability_path)
        except pd.errors.EmptyDataError:
            warnings_table = pd.DataFrame()

        if not warnings_table.empty:
            print("\nIdentifiability warnings:")
            print(warnings_table.to_string(index=False))

    print(f"\nWrote bootstrap outputs to: {output_path}")

    print("\nWritten files:")
    for name, path in written_files.items():
        print(f"  {name}: {path}")


def command_compare_global_observables_multispecies_variable_projection(
    args: argparse.Namespace,
    config: dict,
) -> None:
    engine_name = str(config.get("engine_name", "reference"))
    data_path = config["data"]
    time_column = config.get("time_column", "time")
    signal_columns = config.get("signal_columns")
    exclude_columns = config.get("exclude_columns")
    output_dir = config["output_dir"]

    observed_species = config.get("observed_species")
    observed_species_by_model = config.get(
        "observed_species_by_model",
        observed_species,
    )

    if observed_species_by_model is None:
        raise ValueError(
            "Multispecies model comparison requires observed_species "
            "or observed_species_by_model."
        )

    sort_by = config.get("sort_by", "bic")

    method = config.get("method", "trf")
    loss = config.get("loss", "linear")
    max_nfev = config.get("max_nfev")
    rtol = config.get("rtol", 1e-6)
    atol = config.get("atol", 1e-9)

    fit_offset = bool(config.get("fit_offset", True))

    variable_projection_backend = str(
        config.get("variable_projection_backend", "numpy")
    )
    variable_projection_method = str(config.get("variable_projection_method", "LSODA"))

    models = build_model_specs_from_comparison_config(config)

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

    dataset, filtering_result = read_wide_observable_dataset_with_filtering(
        file_path=data_path,
        time_column=time_column,
        signal_columns=signal_columns,
        exclude_columns=exclude_columns,
        max_missing_fraction=peak_filtering_settings["max_missing_fraction"],
        min_initial_intensity=peak_filtering_settings["min_initial_intensity"],
        initial_points=peak_filtering_settings["initial_points"],
        min_dynamic_range=peak_filtering_settings["min_dynamic_range"],
        interpolate_missing=peak_filtering_settings["interpolate_missing"],
    )

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

    signal_weight_entries = config.get("signal_weight")
    if signal_weight_entries is None:
        signal_weight_entries = config.get("signal_weights")

    settings = FitSettings(
        species_mapping={},
        use_normalized_data=False,
        method=method,
        loss=loss,
        max_nfev=max_nfev,
        rtol=rtol,
        atol=atol,
        signal_weights=parse_signal_weight_entries(signal_weight_entries),
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Running multispecies variable projection model comparison")
    print("Models:", ", ".join(models))
    print("Observable columns:", len(dataset.signal_columns))
    print("Kept observable columns:", len(filtering_result.kept_columns))
    print("Removed observable columns:", len(filtering_result.removed_columns))
    print("Sort by:", sort_by)
    print("Backend:", variable_projection_backend)
    print("ODE method:", variable_projection_method)

    result = fit_global_observable_multispecies_variable_projection_model_comparison(
        models=models,
        dataset=dataset,
        parameter_specs_by_model=parameter_specs_by_model,
        initial_condition_specs_by_model=initial_condition_specs_by_model,
        observed_species_by_model=observed_species_by_model,
        settings=settings,
        signal_columns=dataset.signal_columns,
        fit_offset=fit_offset,
        backend=variable_projection_backend,
        method=variable_projection_method,
        sort_by=sort_by,
        engine_name=engine_name,
    )

    written_files = export_multispecies_variable_projection_model_comparison(
        result=result,
        output_dir=output_path,
        export_best_fit=True,
        export_per_model_fits=False,
    )

    peak_filtering_path = write_peak_filtering_table(
        filtering_result=filtering_result,
        output_dir=output_path,
    )

    written_files["peak_filtering"] = peak_filtering_path

    print("\nBest model:", result.best_model_name)
    print("Best fit success:", result.best_result.success)
    print("Best fit message:", result.best_result.message)
    print("Best fitted kinetic parameters:", result.best_result.fitted_parameters)
    print("Best statistics:", result.best_result.statistics)

    if result.failures:
        print("\nFailed models:")
        for failure in result.failures:
            print(
                f"  {failure.model_name}: {failure.error_type}: {failure.error_message}"
            )

    print(
        "\nWrote multispecies variable projection model comparison outputs "
        f"to: {output_path}"
    )

    print("\nWritten files:")
    for name, path in written_files.items():
        print(f"  {name}: {path}")


def command_multistart_compare_global_observables_multispecies_variable_projection(
    args: argparse.Namespace,
    config: dict,
) -> None:
    engine_name = str(config.get("engine_name", "reference"))
    data_path = config["data"]
    time_column = config.get("time_column", "time")
    signal_columns = config.get("signal_columns")
    exclude_columns = config.get("exclude_columns")
    output_dir = config["output_dir"]

    observed_species = config.get("observed_species")
    observed_species_by_model = config.get(
        "observed_species_by_model",
        observed_species,
    )

    if observed_species_by_model is None:
        raise ValueError(
            "Multispecies multistart model comparison requires observed_species "
            "or observed_species_by_model."
        )

    sort_by = config.get("sort_by", "bic")
    multistart_sort_by = config.get("multistart_sort_by", sort_by)

    n_starts = int(config.get("n_starts", 10))

    random_seed = config.get("random_seed")
    if random_seed is not None:
        random_seed = int(random_seed)

    log_uniform = bool(config.get("log_uniform", True))

    method = config.get("method", "trf")
    loss = config.get("loss", "linear")
    max_nfev = config.get("max_nfev")
    rtol = config.get("rtol", 1e-6)
    atol = config.get("atol", 1e-9)

    fit_offset = bool(config.get("fit_offset", True))

    variable_projection_backend = str(
        config.get("variable_projection_backend", "numpy")
    )
    variable_projection_method = str(config.get("variable_projection_method", "LSODA"))

    show_progress = bool(config.get("show_progress", True)) and not bool(
        getattr(args, "no_progress", False)
    )

    models = build_model_specs_from_comparison_config(config)

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

    dataset, filtering_result = read_wide_observable_dataset_with_filtering(
        file_path=data_path,
        time_column=time_column,
        signal_columns=signal_columns,
        exclude_columns=exclude_columns,
        max_missing_fraction=peak_filtering_settings["max_missing_fraction"],
        min_initial_intensity=peak_filtering_settings["min_initial_intensity"],
        initial_points=peak_filtering_settings["initial_points"],
        min_dynamic_range=peak_filtering_settings["min_dynamic_range"],
        interpolate_missing=peak_filtering_settings["interpolate_missing"],
    )

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

    signal_weight_entries = config.get("signal_weight")
    if signal_weight_entries is None:
        signal_weight_entries = config.get("signal_weights")

    settings = FitSettings(
        species_mapping={},
        use_normalized_data=False,
        method=method,
        loss=loss,
        max_nfev=max_nfev,
        rtol=rtol,
        atol=atol,
        signal_weights=parse_signal_weight_entries(signal_weight_entries),
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Running multispecies variable projection multistart model comparison")
    print("Models:", ", ".join(models))
    print("Observable columns:", len(dataset.signal_columns))
    print("Kept observable columns:", len(filtering_result.kept_columns))
    print("Removed observable columns:", len(filtering_result.removed_columns))
    print("Starts per model:", n_starts)
    print("Model ranking sort by:", sort_by)
    print("Within-model multistart sort by:", multistart_sort_by)
    print("Backend:", variable_projection_backend)
    print("ODE method:", variable_projection_method)

    result = fit_global_observable_multispecies_variable_projection_multistart_model_comparison(
        models=models,
        dataset=dataset,
        parameter_specs_by_model=parameter_specs_by_model,
        initial_condition_specs_by_model=initial_condition_specs_by_model,
        observed_species_by_model=observed_species_by_model,
        settings=settings,
        signal_columns=dataset.signal_columns,
        fit_offset=fit_offset,
        backend=variable_projection_backend,
        method=variable_projection_method,
        n_starts=n_starts,
        random_seed=random_seed,
        sort_by=sort_by,
        multistart_sort_by=multistart_sort_by,
        log_uniform=log_uniform,
        show_progress=show_progress,
        engine_name=engine_name,
    )

    written_files = export_multispecies_variable_projection_multistart_model_comparison(
        result=result,
        output_dir=output_path,
        export_best_fit=True,
        export_per_model_summaries=True,
    )

    peak_filtering_path = write_peak_filtering_table(
        filtering_result=filtering_result,
        output_dir=output_path,
    )

    written_files["peak_filtering"] = peak_filtering_path

    print("\nBest model:", result.best_model_name)
    print("Best model best start:", result.best_result.best_index)
    print("Best fit success:", result.best_fit_result.success)
    print("Best fit message:", result.best_fit_result.message)
    print("Best fitted kinetic parameters:", result.best_fit_result.fitted_parameters)
    print("Best statistics:", result.best_fit_result.statistics)

    if result.failures:
        print("\nFailed models:")
        for failure in result.failures:
            print(
                f"  {failure.model_name}: {failure.error_type}: {failure.error_message}"
            )

    print(
        "\nWrote multispecies variable projection multistart "
        f"model comparison outputs to: {output_path}"
    )

    print("\nWritten files:")
    for name, path in written_files.items():
        print(f"  {name}: {path}")


def command_bootstrap_global_observables_multispecies_variable_projection(
    args: argparse.Namespace,
    config: dict,
) -> None:
    engine_name = str(config.get("engine_name", "reference"))
    model_path = config["model"]
    data_path = config["data"]
    time_column = config.get("time_column", "time")
    signal_columns = config.get("signal_columns")
    exclude_columns = config.get("exclude_columns")
    observed_species = config["observed_species"]
    output_dir = config["output_dir"]

    model = read_model_file(model_path)

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

    dataset, filtering_result = read_wide_observable_dataset_with_filtering(
        file_path=data_path,
        time_column=time_column,
        signal_columns=signal_columns,
        exclude_columns=exclude_columns,
        max_missing_fraction=peak_filtering_settings["max_missing_fraction"],
        min_initial_intensity=peak_filtering_settings["min_initial_intensity"],
        initial_points=peak_filtering_settings["initial_points"],
        min_dynamic_range=peak_filtering_settings["min_dynamic_range"],
        interpolate_missing=peak_filtering_settings["interpolate_missing"],
    )

    parameter_entries = config.get("parameter")
    if parameter_entries is None:
        parameter_entries = config.get("parameters")

    initial_entries = config.get("initial")
    if initial_entries is None:
        initial_entries = config.get("initial_conditions")

    signal_weight_entries = config.get("signal_weight")
    if signal_weight_entries is None:
        signal_weight_entries = config.get("signal_weights")

    parameter_specs = build_parameter_specs(
        model=model,
        parameter_entries=parameter_entries,
        default_guess=float(config.get("default_parameter_guess", 0.1)),
        default_lower=float(config.get("default_parameter_lower", 0.0)),
        default_upper=float(config.get("default_parameter_upper", 100.0)),
    )

    initial_condition_specs = build_initial_condition_specs(
        model=model,
        initial_entries=initial_entries,
    )

    settings = FitSettings(
        species_mapping={},
        use_normalized_data=False,
        method=config.get("method", "trf"),
        loss=config.get("loss", "linear"),
        max_nfev=config.get("max_nfev"),
        rtol=config.get("rtol", 1e-6),
        atol=config.get("atol", 1e-9),
        signal_weights=parse_signal_weight_entries(signal_weight_entries),
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Running multispecies variable projection bootstrap")
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Observed species: {observed_species}")
    print(f"Observable columns: {len(dataset.signal_columns)}")
    print(f"Bootstrap replicates: {int(config.get('n_bootstrap', 100))}")
    print(f"Workers: {int(config.get('n_workers', 1))}")

    result = bootstrap_global_observable_multispecies_variable_projection_fit(
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        observed_species=observed_species,
        settings=settings,
        signal_columns=dataset.signal_columns,
        fit_offset=bool(config.get("fit_offset", True)),
        backend=str(config.get("variable_projection_backend", "numpy")),
        method=str(config.get("variable_projection_method", "LSODA")),
        n_bootstrap=int(config.get("n_bootstrap", 100)),
        n_workers=int(config.get("n_workers", 1)),
        random_seed=config.get("random_seed"),
        confidence_level=float(config.get("confidence_level", 0.95)),
        show_progress=bool(config.get("show_progress", True))
        and not bool(getattr(args, "no_progress", False)),
        engine_name=engine_name,
    )

    written_files = export_multispecies_variable_projection_bootstrap_result(
        result=result,
        output_dir=output_path,
        export_original_fit=True,
        include_plots=not bool(config.get("no_plots", False)),
    )

    peak_filtering_path = write_peak_filtering_table(
        filtering_result=filtering_result,
        output_dir=output_path,
    )

    written_files["peak_filtering"] = peak_filtering_path

    print("\nBootstrap successful fits:", len(result.bootstrap_results))
    print("Bootstrap failed fits:", len(result.failures))
    print("\nParameter uncertainty summary:")
    print(result.summary_table.to_string(index=False))

    print(f"\nWrote multispecies bootstrap outputs to: {output_path}")

    print("\nWritten files:")
    for name, path in written_files.items():
        print(f"  {name}: {path}")


def command_profile_likelihood_global_observables_multispecies_variable_projection(
    args: argparse.Namespace,
    config: dict,
) -> None:
    engine_name = str(config.get("engine_name", "reference"))
    model_path = config["model"]
    data_path = config["data"]
    time_column = config.get("time_column", "time")
    signal_columns = config.get("signal_columns")
    exclude_columns = config.get("exclude_columns")
    observed_species = config["observed_species"]
    output_dir = config["output_dir"]

    model = read_model_file(model_path)

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

    dataset, filtering_result = read_wide_observable_dataset_with_filtering(
        file_path=data_path,
        time_column=time_column,
        signal_columns=signal_columns,
        exclude_columns=exclude_columns,
        max_missing_fraction=peak_filtering_settings["max_missing_fraction"],
        min_initial_intensity=peak_filtering_settings["min_initial_intensity"],
        initial_points=peak_filtering_settings["initial_points"],
        min_dynamic_range=peak_filtering_settings["min_dynamic_range"],
        interpolate_missing=peak_filtering_settings["interpolate_missing"],
    )

    parameter_entries = config.get("parameter")
    if parameter_entries is None:
        parameter_entries = config.get("parameters")

    initial_entries = config.get("initial")
    if initial_entries is None:
        initial_entries = config.get("initial_conditions")

    signal_weight_entries = config.get("signal_weight")
    if signal_weight_entries is None:
        signal_weight_entries = config.get("signal_weights")

    parameter_specs = build_parameter_specs(
        model=model,
        parameter_entries=parameter_entries,
        default_guess=float(config.get("default_parameter_guess", 0.1)),
        default_lower=float(config.get("default_parameter_lower", 0.0)),
        default_upper=float(config.get("default_parameter_upper", 100.0)),
    )

    initial_condition_specs = build_initial_condition_specs(
        model=model,
        initial_entries=initial_entries,
    )

    settings = FitSettings(
        species_mapping={},
        use_normalized_data=False,
        method=config.get("method", "trf"),
        loss=config.get("loss", "linear"),
        max_nfev=config.get("max_nfev"),
        rtol=config.get("rtol", 1e-6),
        atol=config.get("atol", 1e-9),
        signal_weights=parse_signal_weight_entries(signal_weight_entries),
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Running multispecies variable projection profile likelihood")
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Observed species: {observed_species}")
    print(f"Observable columns: {len(dataset.signal_columns)}")

    result = fit_multispecies_variable_projection_profile_likelihood(
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        observed_species=observed_species,
        settings=settings,
        signal_columns=dataset.signal_columns,
        fit_offset=bool(config.get("fit_offset", True)),
        backend=str(config.get("variable_projection_backend", "numpy")),
        method=str(config.get("variable_projection_method", "LSODA")),
        profile_parameters=config.get("profile_parameters"),
        n_points=int(config.get("profile_n_points", 15)),
        span_factor=float(config.get("profile_span_factor", 10.0)),
        log_space=bool(config.get("profile_log_space", True)),
        show_progress=not bool(getattr(args, "no_progress", False)),
        engine_name=engine_name,
    )

    written_files = export_multispecies_profile_likelihood_result(
        result=result,
        output_dir=output_path,
        include_plots=not bool(config.get("no_plots", False)),
    )

    peak_filtering_path = write_peak_filtering_table(
        filtering_result=filtering_result,
        output_dir=output_path,
    )

    written_files["peak_filtering"] = peak_filtering_path

    print("\nMultispecies profile likelihood complete")
    print(result.profile_table.to_string(index=False))
    print(f"\nWrote multispecies profile likelihood outputs to: {output_path}")

    print("\nWritten files:")
    for name, path in written_files.items():
        print(f"  {name}: {path}")


def build_parser() -> argparse.ArgumentParser:
    """
    Build CLI argument parser.
    """

    parser = argparse.ArgumentParser(
        prog="odefit",
        description="Build, simulate, fit, and export ODE models.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    generate_parser = subparsers.add_parser(
        "generate-odes",
        help="Generate ODEs from a reaction model text file.",
    )

    generate_parser.add_argument(
        "--model",
        required=True,
        help="Path to model text file.",
    )

    generate_parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write generated ODEs.",
    )

    generate_parser.set_defaults(func=command_generate_odes)

    simulate_parser = subparsers.add_parser(
        "simulate",
        help="Simulate a reaction model and export simulated curves.",
    )

    simulate_parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON simulation configuration file.",
    )

    simulate_parser.add_argument(
        "--model",
        default=None,
        help="Path to model text file.",
    )

    simulate_parser.add_argument(
        "--parameter-value",
        action="append",
        default=None,
        help="Parameter value: name:value. Example: k1f:0.5. Can be repeated.",
    )

    simulate_parser.add_argument(
        "--initial-value",
        action="append",
        default=None,
        help="Initial condition value: species:value. Example: A:1.0. Can be repeated.",
    )

    simulate_parser.add_argument(
        "--timepoints",
        nargs="+",
        default=None,
        help="Explicit simulation timepoints.",
    )

    simulate_parser.add_argument(
        "--time-start",
        type=float,
        default=None,
        help="Simulation start time.",
    )

    simulate_parser.add_argument(
        "--time-end",
        type=float,
        default=None,
        help="Simulation end time.",
    )

    simulate_parser.add_argument(
        "--num-points",
        type=int,
        default=None,
        help="Number of simulation timepoints.",
    )

    simulate_parser.add_argument(
        "--method",
        default=None,
        help="solve_ivp method, e.g. LSODA, RK45, BDF, Radau.",
    )

    simulate_parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="ODE solver relative tolerance.",
    )

    simulate_parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="ODE solver absolute tolerance.",
    )

    simulate_parser.add_argument(
        "--clip-negative-concentrations",
        action="store_true",
        help="Clip negative concentrations to zero inside RHS evaluation.",
    )

    simulate_parser.add_argument(
        "--no-negative-warnings",
        action="store_true",
        help="Do not report warnings for negative simulated values.",
    )

    simulate_parser.add_argument(
        "--output-csv",
        default=None,
        help="Path to write simulated curves CSV.",
    )

    simulate_parser.add_argument(
        "--output-plot",
        default=None,
        help="Optional path to write simulation plot image.",
    )

    simulate_parser.set_defaults(func=command_simulate)

    fit_parser = subparsers.add_parser(
        "fit",
        help="Fit a reaction model to a CSV dataset.",
    )

    fit_parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON fit configuration file.",
    )

    fit_parser.add_argument(
        "--model",
        default=None,
        help="Path to model text file.",
    )

    fit_parser.add_argument(
        "--data",
        default=None,
        help="Path to CSV data file.",
    )

    fit_parser.add_argument(
        "--time-column",
        default=None,
        help="Name of the time column.",
    )

    fit_parser.add_argument(
        "--signal-columns",
        nargs="+",
        default=None,
        help="Signal/data columns to fit.",
    )

    fit_parser.add_argument(
        "--mapping",
        action="append",
        default=None,
        help="Data-column to species mapping: data_column:species. Can be repeated.",
    )

    fit_parser.add_argument(
        "--parameter",
        action="append",
        default=None,
        help=(
            "Parameter override: name:initial_guess:lower_bound:upper_bound. "
            "Can be repeated."
        ),
    )

    fit_parser.add_argument(
        "--initial",
        action="append",
        default=None,
        help=(
            "Initial condition: species:value:fixed_or_fit:lower_bound:upper_bound. "
            "Example: A:1.0:fixed:0:10. Can be repeated."
        ),
    )

    fit_parser.add_argument(
        "--signal-weight",
        action="append",
        default=None,
        help="Signal residual weight: data_column:weight. Can be repeated.",
    )

    fit_parser.add_argument(
        "--default-parameter-guess",
        type=float,
        default=None,
        help="Default initial guess for model parameters.",
    )

    fit_parser.add_argument(
        "--default-parameter-lower",
        type=float,
        default=None,
        help="Default lower bound for model parameters.",
    )

    fit_parser.add_argument(
        "--default-parameter-upper",
        type=float,
        default=None,
        help="Default upper bound for model parameters.",
    )

    fit_parser.add_argument(
        "--method",
        default=None,
        help="scipy.optimize.least_squares method.",
    )

    fit_parser.add_argument(
        "--loss",
        default=None,
        help="scipy.optimize.least_squares loss.",
    )

    fit_parser.add_argument(
        "--max-nfev",
        type=int,
        default=None,
        help="Maximum number of function evaluations.",
    )

    fit_parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="ODE solver relative tolerance.",
    )

    fit_parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="ODE solver absolute tolerance.",
    )

    fit_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for fit bundle.",
    )

    fit_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )

    fit_parser.set_defaults(func=command_fit)

    amyloid_parser = subparsers.add_parser(
        "fit-amyloid-aggregation",
        help="Fit the built-in amyloid aggregation model across conditions.",
    )

    amyloid_parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON amyloid aggregation fit config.",
    )

    amyloid_parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory overriding config output_dir.",
    )

    amyloid_parser.set_defaults(func=command_fit_amyloid_aggregation)

    multistart_parser = subparsers.add_parser(
        "multistart",
        help="Run multistart fitting from a CSV dataset or JSON config.",
    )

    multistart_parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON fit configuration file.",
    )

    multistart_parser.add_argument(
        "--model",
        default=None,
        help="Path to model text file.",
    )

    multistart_parser.add_argument(
        "--data",
        default=None,
        help="Path to CSV data file.",
    )

    multistart_parser.add_argument(
        "--time-column",
        default=None,
        help="Name of the time column.",
    )

    multistart_parser.add_argument(
        "--signal-columns",
        nargs="+",
        default=None,
        help="Signal/data columns to fit.",
    )

    multistart_parser.add_argument(
        "--mapping",
        action="append",
        default=None,
        help="Data-column to species mapping: data_column:species. Can be repeated.",
    )

    multistart_parser.add_argument(
        "--parameter",
        action="append",
        default=None,
        help=(
            "Parameter override: name:initial_guess:lower_bound:upper_bound. "
            "Can be repeated."
        ),
    )

    multistart_parser.add_argument(
        "--initial",
        action="append",
        default=None,
        help=(
            "Initial condition: species:value:fixed_or_fit:lower_bound:upper_bound. "
            "Example: A:1.0:fixed:0:10. Can be repeated."
        ),
    )

    multistart_parser.add_argument(
        "--signal-weight",
        action="append",
        default=None,
        help="Signal residual weight: data_column:weight. Can be repeated.",
    )

    multistart_parser.add_argument(
        "--default-parameter-guess",
        type=float,
        default=None,
        help="Default initial guess for model parameters.",
    )

    multistart_parser.add_argument(
        "--default-parameter-lower",
        type=float,
        default=None,
        help="Default lower bound for model parameters.",
    )

    multistart_parser.add_argument(
        "--default-parameter-upper",
        type=float,
        default=None,
        help="Default upper bound for model parameters.",
    )

    multistart_parser.add_argument(
        "--method",
        default=None,
        help="scipy.optimize.least_squares method.",
    )

    multistart_parser.add_argument(
        "--loss",
        default=None,
        help="scipy.optimize.least_squares loss.",
    )

    multistart_parser.add_argument(
        "--max-nfev",
        type=int,
        default=None,
        help="Maximum number of function evaluations.",
    )

    multistart_parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="ODE solver relative tolerance.",
    )

    multistart_parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="ODE solver absolute tolerance.",
    )

    multistart_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for multistart bundle.",
    )

    multistart_parser.add_argument(
        "--n-starts",
        type=int,
        default=None,
        help="Number of multistart fits.",
    )

    multistart_parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of parallel worker processes. Use 1 for serial.",
    )

    multistart_parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for generated starting guesses.",
    )

    multistart_parser.add_argument(
        "--sort-by",
        default=None,
        help="Metric used to rank starts. Usually aic, bic, rmse, rss, or cost.",
    )

    multistart_parser.add_argument(
        "--linear-sampling",
        action="store_true",
        help="Use linear rather than log-uniform sampling for starting guesses.",
    )

    multistart_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation for best fit bundle.",
    )

    multistart_parser.set_defaults(func=command_multistart)

    global_observable_parser = subparsers.add_parser(
        "fit-global-observables",
        help="Fit many observable columns to one shared kinetic model.",
    )

    global_observable_parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON global observable fitting config.",
    )

    global_observable_parser.add_argument(
        "--model",
        default=None,
        help="Path to model text file.",
    )

    global_observable_parser.add_argument(
        "--data",
        default=None,
        help="Path to wide-format observable CSV file.",
    )

    global_observable_parser.add_argument(
        "--time-column",
        default=None,
        help="Name of the time column.",
    )

    global_observable_parser.add_argument(
        "--signal-columns",
        nargs="+",
        default=None,
        help="Observable/signal columns. If omitted, numeric columns are inferred.",
    )

    global_observable_parser.add_argument(
        "--exclude-columns",
        nargs="+",
        default=None,
        help="Columns to exclude when inferring signal columns.",
    )

    global_observable_parser.add_argument(
        "--observed-species",
        default=None,
        help="Model species that all observables map to. Default: A.",
    )

    global_observable_parser.add_argument(
        "--parameter",
        action="append",
        default=None,
        help=(
            "Parameter override: name:initial_guess:lower_bound:upper_bound. "
            "Can be repeated."
        ),
    )

    global_observable_parser.add_argument(
        "--initial",
        action="append",
        default=None,
        help=(
            "Initial condition: species:value:fixed_or_fit:lower_bound:upper_bound. "
            "Can be repeated."
        ),
    )

    global_observable_parser.add_argument(
        "--signal-weight",
        action="append",
        default=None,
        help="Signal residual weight: data_column:weight. Can be repeated.",
    )

    global_observable_parser.add_argument(
        "--default-parameter-guess",
        type=float,
        default=None,
        help="Default initial guess for model parameters.",
    )

    global_observable_parser.add_argument(
        "--default-parameter-lower",
        type=float,
        default=None,
        help="Default lower bound for model parameters.",
    )

    global_observable_parser.add_argument(
        "--default-parameter-upper",
        type=float,
        default=None,
        help="Default upper bound for model parameters.",
    )

    global_observable_parser.add_argument(
        "--method",
        default=None,
        help="scipy.optimize.least_squares method.",
    )

    global_observable_parser.add_argument(
        "--loss",
        default=None,
        help="scipy.optimize.least_squares loss.",
    )

    global_observable_parser.add_argument(
        "--max-nfev",
        type=int,
        default=None,
        help="Maximum number of function evaluations.",
    )

    global_observable_parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="ODE solver relative tolerance.",
    )

    global_observable_parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="ODE solver absolute tolerance.",
    )

    global_observable_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for global observable fit bundle.",
    )

    global_observable_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )

    global_observable_parser.add_argument(
        "--max-missing-fraction",
        type=float,
        default=None,
        help="Remove signal columns with missing fraction above this value.",
    )

    global_observable_parser.add_argument(
        "--min-initial-intensity",
        type=float,
        default=None,
        help="Remove signal columns with initial intensity below this value.",
    )

    global_observable_parser.add_argument(
        "--initial-points",
        type=int,
        default=None,
        help="Number of initial timepoints used to estimate initial intensity.",
    )

    global_observable_parser.add_argument(
        "--min-dynamic-range",
        type=float,
        default=None,
        help="Remove signal columns with dynamic range below this value.",
    )

    global_observable_parser.add_argument(
        "--no-interpolate-missing",
        action="store_true",
        help="Do not interpolate missing values in kept signal columns.",
    )

    global_observable_parser.add_argument(
        "--variable-projection",
        action="store_true",
        help=(
            "Use variable projection for shared-species global observable fitting. "
            "This analytically solves per-column scale/offset terms."
        ),
    )

    global_observable_parser.set_defaults(func=command_fit_global_observables)

    global_observable_multistart_parser = subparsers.add_parser(
        "multistart-global-observables",
        help="Run multistart fitting for many observable columns sharing one kinetic model.",
    )

    global_observable_multistart_parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON global observable multistart config.",
    )

    global_observable_multistart_parser.add_argument(
        "--model",
        default=None,
        help="Path to model text file.",
    )

    global_observable_multistart_parser.add_argument(
        "--data",
        default=None,
        help="Path to wide-format observable CSV file.",
    )

    global_observable_multistart_parser.add_argument(
        "--time-column",
        default=None,
        help="Name of the time column.",
    )

    global_observable_multistart_parser.add_argument(
        "--signal-columns",
        nargs="+",
        default=None,
        help="Observable/signal columns. If omitted, numeric columns are inferred.",
    )

    global_observable_multistart_parser.add_argument(
        "--exclude-columns",
        nargs="+",
        default=None,
        help="Columns to exclude when inferring signal columns.",
    )

    global_observable_multistart_parser.add_argument(
        "--observed-species",
        default=None,
        help="Model species that all observables map to. Default: A.",
    )

    global_observable_multistart_parser.add_argument(
        "--parameter",
        action="append",
        default=None,
        help=(
            "Parameter override: name:initial_guess:lower_bound:upper_bound. "
            "Can be repeated."
        ),
    )

    global_observable_multistart_parser.add_argument(
        "--initial",
        action="append",
        default=None,
        help=(
            "Initial condition: species:value:fixed_or_fit:lower_bound:upper_bound. "
            "Can be repeated."
        ),
    )

    global_observable_multistart_parser.add_argument(
        "--signal-weight",
        action="append",
        default=None,
        help="Signal residual weight: data_column:weight. Can be repeated.",
    )

    global_observable_multistart_parser.add_argument(
        "--default-parameter-guess",
        type=float,
        default=None,
        help="Default initial guess for model parameters.",
    )

    global_observable_multistart_parser.add_argument(
        "--default-parameter-lower",
        type=float,
        default=None,
        help="Default lower bound for model parameters.",
    )

    global_observable_multistart_parser.add_argument(
        "--default-parameter-upper",
        type=float,
        default=None,
        help="Default upper bound for model parameters.",
    )

    global_observable_multistart_parser.add_argument(
        "--method",
        default=None,
        help="scipy.optimize.least_squares method.",
    )

    global_observable_multistart_parser.add_argument(
        "--loss",
        default=None,
        help="scipy.optimize.least_squares loss.",
    )

    global_observable_multistart_parser.add_argument(
        "--max-nfev",
        type=int,
        default=None,
        help="Maximum number of function evaluations.",
    )

    global_observable_multistart_parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="ODE solver relative tolerance.",
    )

    global_observable_multistart_parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="ODE solver absolute tolerance.",
    )

    global_observable_multistart_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for global observable multistart outputs.",
    )

    global_observable_multistart_parser.add_argument(
        "--n-starts",
        type=int,
        default=None,
        help="Number of multistart fits.",
    )

    global_observable_multistart_parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of parallel worker processes. Use 1 for serial.",
    )

    global_observable_multistart_parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for generated starting guesses.",
    )

    global_observable_multistart_parser.add_argument(
        "--sort-by",
        default=None,
        help="Metric used to rank starts. Usually aic, bic, rmse, rss, or cost.",
    )

    global_observable_multistart_parser.add_argument(
        "--linear-sampling",
        action="store_true",
        help="Use linear rather than log-uniform sampling for starting guesses.",
    )

    global_observable_multistart_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation for best fit bundle.",
    )
    global_observable_multistart_parser.add_argument(
        "--max-missing-fraction",
        type=float,
        default=None,
        help="Remove signal columns with missing fraction above this value.",
    )

    global_observable_multistart_parser.add_argument(
        "--min-initial-intensity",
        type=float,
        default=None,
        help="Remove signal columns with initial intensity below this value.",
    )

    global_observable_multistart_parser.add_argument(
        "--initial-points",
        type=int,
        default=None,
        help="Number of initial timepoints used to estimate initial intensity.",
    )

    global_observable_multistart_parser.add_argument(
        "--min-dynamic-range",
        type=float,
        default=None,
        help="Remove signal columns with dynamic range below this value.",
    )

    global_observable_multistart_parser.add_argument(
        "--no-interpolate-missing",
        action="store_true",
        help="Do not interpolate missing values in kept signal columns.",
    )

    global_observable_multistart_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress and ETA output.",
    )

    global_observable_multistart_parser.add_argument(
        "--variable-projection",
        action="store_true",
        help=(
            "Use variable projection for shared-species global observable "
            "multistart fitting. This analytically solves per-column "
            "scale/offset terms."
        ),
    )

    global_observable_multistart_parser.set_defaults(
        func=command_multistart_global_observables
    )
    compare_global_observable_parser = subparsers.add_parser(
        "compare-global-observables",
        help="Compare several global observable mechanisms on one dataset.",
    )

    compare_global_observable_parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON global observable model comparison config.",
    )

    compare_global_observable_parser.add_argument(
        "--data",
        default=None,
        help="Path to wide-format observable CSV file.",
    )

    compare_global_observable_parser.add_argument(
        "--time-column",
        default=None,
        help="Name of the time column.",
    )

    compare_global_observable_parser.add_argument(
        "--signal-columns",
        nargs="+",
        default=None,
        help="Observable/signal columns. If omitted, numeric columns are inferred.",
    )

    compare_global_observable_parser.add_argument(
        "--exclude-columns",
        nargs="+",
        default=None,
        help="Columns to exclude when inferring signal columns.",
    )

    compare_global_observable_parser.add_argument(
        "--observed-species",
        default=None,
        help="Default observed model species for all models. Default: A.",
    )

    compare_global_observable_parser.add_argument(
        "--sort-by",
        default=None,
        help="Metric used to rank models. Usually aic, bic, rmse, rss, or cost.",
    )

    compare_global_observable_parser.add_argument(
        "--default-parameter-guess",
        type=float,
        default=None,
        help="Default initial guess for model parameters.",
    )

    compare_global_observable_parser.add_argument(
        "--default-parameter-lower",
        type=float,
        default=None,
        help="Default lower bound for model parameters.",
    )

    compare_global_observable_parser.add_argument(
        "--default-parameter-upper",
        type=float,
        default=None,
        help="Default upper bound for model parameters.",
    )

    compare_global_observable_parser.add_argument(
        "--method",
        default=None,
        help="scipy.optimize.least_squares method.",
    )

    compare_global_observable_parser.add_argument(
        "--loss",
        default=None,
        help="scipy.optimize.least_squares loss.",
    )

    compare_global_observable_parser.add_argument(
        "--max-nfev",
        type=int,
        default=None,
        help="Maximum number of function evaluations.",
    )

    compare_global_observable_parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="ODE solver relative tolerance.",
    )

    compare_global_observable_parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="ODE solver absolute tolerance.",
    )

    compare_global_observable_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for model comparison outputs.",
    )

    compare_global_observable_parser.add_argument(
        "--max-missing-fraction",
        type=float,
        default=None,
        help="Remove signal columns with missing fraction above this value.",
    )

    compare_global_observable_parser.add_argument(
        "--min-initial-intensity",
        type=float,
        default=None,
        help="Remove signal columns with initial intensity below this value.",
    )

    compare_global_observable_parser.add_argument(
        "--initial-points",
        type=int,
        default=None,
        help="Number of initial timepoints used to estimate initial intensity.",
    )

    compare_global_observable_parser.add_argument(
        "--min-dynamic-range",
        type=float,
        default=None,
        help="Remove signal columns with dynamic range below this value.",
    )

    compare_global_observable_parser.add_argument(
        "--no-interpolate-missing",
        action="store_true",
        help="Do not interpolate missing values in kept signal columns.",
    )

    compare_global_observable_parser.add_argument(
        "--signal-weight",
        action="append",
        default=None,
        help="Signal residual weight: data_column:weight. Can be repeated.",
    )

    compare_global_observable_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation for best fit bundle.",
    )
    compare_global_observable_parser.add_argument(
        "--variable-projection",
        action="store_true",
        help=(
            "Use variable projection for shared-species global observable "
            "model comparison. This analytically solves per-column "
            "scale/offset terms."
        ),
    )

    compare_global_observable_parser.set_defaults(
        func=command_compare_global_observables
    )
    multistart_compare_global_observable_parser = subparsers.add_parser(
        "multistart-compare-global-observables",
        help="Compare several global observable mechanisms using multistart per model.",
    )

    multistart_compare_global_observable_parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON multistart model comparison config.",
    )

    multistart_compare_global_observable_parser.add_argument(
        "--data",
        default=None,
        help="Path to wide-format observable CSV file.",
    )

    multistart_compare_global_observable_parser.add_argument(
        "--time-column",
        default=None,
        help="Name of the time column.",
    )

    multistart_compare_global_observable_parser.add_argument(
        "--signal-columns",
        nargs="+",
        default=None,
        help="Observable/signal columns. If omitted, numeric columns are inferred.",
    )

    multistart_compare_global_observable_parser.add_argument(
        "--exclude-columns",
        nargs="+",
        default=None,
        help="Columns to exclude when inferring signal columns.",
    )

    multistart_compare_global_observable_parser.add_argument(
        "--observed-species",
        default=None,
        help="Default observed model species for all models. Default: A.",
    )

    multistart_compare_global_observable_parser.add_argument(
        "--sort-by",
        default=None,
        help="Metric used to rank models. Usually aic, bic, rmse, rss, or cost.",
    )

    multistart_compare_global_observable_parser.add_argument(
        "--default-parameter-guess",
        type=float,
        default=None,
        help="Default initial guess for model parameters.",
    )

    multistart_compare_global_observable_parser.add_argument(
        "--default-parameter-lower",
        type=float,
        default=None,
        help="Default lower bound for model parameters.",
    )

    multistart_compare_global_observable_parser.add_argument(
        "--default-parameter-upper",
        type=float,
        default=None,
        help="Default upper bound for model parameters.",
    )

    multistart_compare_global_observable_parser.add_argument(
        "--method",
        default=None,
        help="scipy.optimize.least_squares method.",
    )

    multistart_compare_global_observable_parser.add_argument(
        "--loss",
        default=None,
        help="scipy.optimize.least_squares loss.",
    )

    multistart_compare_global_observable_parser.add_argument(
        "--max-nfev",
        type=int,
        default=None,
        help="Maximum number of function evaluations.",
    )

    multistart_compare_global_observable_parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="ODE solver relative tolerance.",
    )

    multistart_compare_global_observable_parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="ODE solver absolute tolerance.",
    )

    multistart_compare_global_observable_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for multistart model comparison outputs.",
    )

    multistart_compare_global_observable_parser.add_argument(
        "--n-starts",
        type=int,
        default=None,
        help="Number of multistart fits per model.",
    )

    multistart_compare_global_observable_parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of worker processes per model. Use 1 for serial.",
    )

    multistart_compare_global_observable_parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for generated starting guesses.",
    )

    multistart_compare_global_observable_parser.add_argument(
        "--linear-sampling",
        action="store_true",
        help="Use linear rather than log-uniform sampling for kinetic parameters.",
    )

    multistart_compare_global_observable_parser.add_argument(
        "--max-missing-fraction",
        type=float,
        default=None,
        help="Remove signal columns with missing fraction above this value.",
    )

    multistart_compare_global_observable_parser.add_argument(
        "--min-initial-intensity",
        type=float,
        default=None,
        help="Remove signal columns with initial intensity below this value.",
    )

    multistart_compare_global_observable_parser.add_argument(
        "--initial-points",
        type=int,
        default=None,
        help="Number of initial timepoints used to estimate initial intensity.",
    )

    multistart_compare_global_observable_parser.add_argument(
        "--min-dynamic-range",
        type=float,
        default=None,
        help="Remove signal columns with dynamic range below this value.",
    )

    multistart_compare_global_observable_parser.add_argument(
        "--no-interpolate-missing",
        action="store_true",
        help="Do not interpolate missing values in kept signal columns.",
    )

    multistart_compare_global_observable_parser.add_argument(
        "--signal-weight",
        action="append",
        default=None,
        help="Signal residual weight: data_column:weight. Can be repeated.",
    )

    multistart_compare_global_observable_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation for best fit bundle.",
    )

    multistart_compare_global_observable_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress and ETA output.",
    )

    multistart_compare_global_observable_parser.add_argument(
        "--variable-projection",
        action="store_true",
        help=(
            "Use variable projection for multistart global observable "
            "model comparison. This analytically solves per-column "
            "scale/offset terms."
        ),
    )

    multistart_compare_global_observable_parser.set_defaults(
        func=command_multistart_compare_global_observables
    )

    bootstrap_global_observables_parser = subparsers.add_parser(
        "bootstrap-global-observables",
        help=("Estimate uncertainty for global observable fits using bootstrap."),
    )

    bootstrap_global_observables_parser.add_argument(
        "--config",
        required=True,
        help="Path to bootstrap configuration JSON.",
    )

    bootstrap_global_observables_parser.add_argument(
        "--variable-projection",
        action="store_true",
        help="Use variable projection bootstrap.",
    )

    bootstrap_global_observables_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable bootstrap progress output.",
    )

    bootstrap_global_observables_parser.set_defaults(
        func=command_bootstrap_global_observables
    )

    profile_likelihood_parser = subparsers.add_parser(
        "profile-likelihood-global-observables",
        help="Run profile likelihood analysis for variable-projection global observable fits.",
    )

    profile_likelihood_parser.add_argument(
        "--config",
        required=True,
    )

    profile_likelihood_parser.add_argument(
        "--variable-projection",
        action="store_true",
    )

    profile_likelihood_parser.add_argument(
        "--no-progress",
        action="store_true",
    )

    profile_likelihood_parser.set_defaults(
        func=command_profile_likelihood_global_observables
    )

    performance_parser = subparsers.add_parser(
        "performance-info",
        help="Show optional performance backend availability.",
    )
    benchmark_parser = subparsers.add_parser(
        "benchmark-performance",
        help="Run small mODEler performance benchmarks.",
    )

    benchmark_parser.add_argument(
        "--output",
        default="benchmarks/performance_benchmarks.csv",
        help="Path to benchmark output CSV.",
    )

    benchmark_parser.set_defaults(func=command_benchmark_performance)

    performance_parser.set_defaults(func=command_performance_info)

    return parser


def main(argv: list[str] | None = None) -> None:
    """
    CLI entry point.
    """

    parser = build_parser()
    args = parser.parse_args(argv)

    args.func(args)


if __name__ == "__main__":
    main()

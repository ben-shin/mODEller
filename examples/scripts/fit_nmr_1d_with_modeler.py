from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from odefit.api.backend import (
    fit_global_observables_from_config,
    get_backend_engine_capabilities,
    parse_model_text,
    validate_backend_engine_name,
)
from odefit.api.serialization import backend_output_payload


def parse_parameter_override(entry: str) -> tuple[str, dict[str, float]]:
    parts = entry.split(":")

    if len(parts) != 4:
        raise ValueError(
            "Parameter overrides must use name:initial_guess:lower_bound:upper_bound. "
            f"Got: {entry}"
        )

    name, initial_guess, lower_bound, upper_bound = parts

    return name, {
        "initial_guess": float(initial_guess),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
    }


def parse_initial_condition_override(entry: str) -> tuple[str, dict[str, Any]]:
    parts = entry.split(":")

    if len(parts) != 5:
        raise ValueError(
            "Initial condition overrides must use "
            "species:value:fixed_or_fit:lower_bound:upper_bound. "
            f"Got: {entry}"
        )

    species, value, mode, lower_bound, upper_bound = parts

    if mode not in {"fixed", "fit"}:
        raise ValueError(f"Initial condition mode must be fixed or fit. Got: {mode}")

    return species, {
        "value": float(value),
        "mode": mode,
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
    }


def numeric_signal_columns(
    dataframe: pd.DataFrame,
    *,
    time_column: str,
    exclude_columns: list[str],
) -> list[str]:
    excluded = set(exclude_columns) | {time_column}

    return [
        column
        for column in dataframe.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(dataframe[column])
    ]


def normalize_signals(
    dataframe: pd.DataFrame,
    *,
    signal_columns: list[str],
    mode: str,
) -> pd.DataFrame:
    if mode == "none":
        return dataframe.copy()

    output = dataframe.copy()

    for column in signal_columns:
        values = output[column].astype(float)

        if mode == "first":
            denominator = values.iloc[0]
        elif mode == "max":
            denominator = values.max()
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")

        if denominator == 0:
            raise ValueError(f"Cannot normalize {column}: denominator is zero")

        output[column] = values / denominator

    return output


def build_default_parameters(parsed_model: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        parameter: {
            "initial_guess": 0.01,
            "lower_bound": 1e-6,
            "upper_bound": 10.0,
        }
        for parameter in parsed_model["parameters"]
    }


def build_default_initial_conditions(parsed_model: dict[str, Any]) -> dict[str, dict[str, Any]]:
    initial_conditions = {}

    for species in parsed_model["species"]:
        initial_conditions[species] = {
            "value": 1.0 if species == "A" else 0.0,
            "mode": "fixed",
            "lower_bound": 0.0,
            "upper_bound": 10.0,
        }

    return initial_conditions


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def run_cli(command: list[str], *, env: dict[str, str]) -> None:
    print("\nRunning mODEler CLI:")
    print(" ".join(command))

    subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )


def build_config(args: argparse.Namespace) -> tuple[dict[str, Any], Path, list[str]]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(args.data)

    signal_columns = args.signal_columns

    if signal_columns is None:
        signal_columns = numeric_signal_columns(
            dataframe,
            time_column=args.time_column,
            exclude_columns=args.exclude_columns or [],
        )

    if not signal_columns:
        raise ValueError("No signal columns were selected or inferred.")

    normalized_dataframe = normalize_signals(
        dataframe,
        signal_columns=signal_columns,
        mode=args.normalize,
    )

    prepared_data_path = output_dir / "prepared_nmr_1d_data.csv"
    normalized_dataframe.to_csv(prepared_data_path, index=False)

    if args.model_file is not None:
        model_path = Path(args.model_file).resolve()
        model_text = model_path.read_text()
    else:
        model_path = output_dir / "model.txt"
        model_text = args.model_text
        model_path.write_text(model_text.strip() + "\n")

    parsed_model = parse_model_text(model_text)

    parameters = build_default_parameters(parsed_model)

    for entry in args.parameter or []:
        name, values = parse_parameter_override(entry)
        parameters[name] = values

    initial_conditions = build_default_initial_conditions(parsed_model)

    for entry in args.initial or []:
        species, values = parse_initial_condition_override(entry)
        initial_conditions[species] = values

    config: dict[str, Any] = {
        "model": str(model_path),
        "data": str(prepared_data_path),
        "time_column": args.time_column,
        "signal_columns": signal_columns,
        "parameters": parameters,
        "initial_conditions": initial_conditions,
        "method": args.optimizer_method,
        "loss": args.loss,
        "max_nfev": args.max_nfev,
        "rtol": args.rtol,
        "atol": args.atol,
        "output_dir": str(output_dir / "fit_output"),
        "engine_name": args.engine_name,
    }

    if args.workflow == "variable-projection":
        config.update(
            {
                "use_variable_projection": True,
                "observed_species": args.observed_species,
                "fit_scale": args.fit_scale,
                "fit_offset": args.fit_offset,
                "variable_projection_backend": args.variable_projection_backend,
                "variable_projection_method": args.ode_method,
                "max_missing_fraction": args.max_missing_fraction,
                "interpolate_missing": True,
            }
        )
    else:
        if len(signal_columns) != 1 and args.mapping is None:
            raise ValueError(
                "Direct fitting with multiple signal columns needs --mapping entries. "
                "Use --workflow variable-projection for many 1D peak/region columns."
            )

        mapping = {
            signal_columns[0]: args.observed_species,
        }

        for entry in args.mapping or []:
            data_column, species = entry.split(":", 1)
            mapping[data_column] = species

        config.update(
            {
                "mapping": mapping,
                "no_plots": args.no_plots,
            }
        )

    return config, output_dir, signal_columns


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit 1D NMR timecourse intensities with mODEler using either "
            "direct species mapping or global observable variable projection."
        )
    )

    parser.add_argument("--data", required=True, help="Input CSV with time and intensities.")
    parser.add_argument("--time-column", default="time")
    parser.add_argument("--signal-columns", nargs="+", default=None)
    parser.add_argument("--exclude-columns", nargs="+", default=None)
    parser.add_argument("--normalize", choices=["none", "first", "max"], default="first")

    parser.add_argument("--model-file", default=None)
    parser.add_argument("--model-text", default="A>B")
    parser.add_argument("--observed-species", default="A")

    parser.add_argument(
        "--workflow",
        choices=["variable-projection", "direct"],
        default="variable-projection",
        help=(
            "variable-projection is recommended for many 1D peaks/regions that "
            "report on the same kinetic species."
        ),
    )
    parser.add_argument("--mapping", action="append", default=None)

    parser.add_argument("--parameter", action="append", default=None)
    parser.add_argument("--initial", action="append", default=None)

    parser.add_argument("--engine-name", default="reference")
    parser.add_argument("--variable-projection-backend", default="numpy")
    parser.add_argument("--ode-method", default="LSODA")
    parser.add_argument("--optimizer-method", default="trf")
    parser.add_argument("--loss", default="linear")
    parser.add_argument("--max-nfev", type=int, default=2000)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--max-missing-fraction", type=float, default=0.0)

    parser.add_argument("--fit-scale", action="store_true", default=True)
    parser.add_argument("--no-fit-scale", action="store_false", dest="fit_scale")
    parser.add_argument("--fit-offset", action="store_true", default=True)
    parser.add_argument("--no-fit-offset", action="store_false", dest="fit_offset")
    parser.add_argument("--no-plots", action="store_true")

    parser.add_argument("--output-dir", default="outputs/nmr_1d_modeler_fit")
    parser.add_argument("--config-name", default="nmr_1d_fit_config.json")
    parser.add_argument("--skip-cli", action="store_true")
    parser.add_argument("--also-run-api-engine", action="store_true")
    parser.add_argument("--payload-rows", type=int, default=50)

    args = parser.parse_args()

    engine_validation = validate_backend_engine_name(args.engine_name)

    if not engine_validation["valid"]:
        raise ValueError(
            f"Invalid mODEler engine {args.engine_name!r}: "
            f"{engine_validation['error_message']}"
        )

    config, output_dir, signal_columns = build_config(args)
    config_path = output_dir / args.config_name
    write_json(config, config_path)

    engine_payload_path = output_dir / "available_engines.json"
    write_json(get_backend_engine_capabilities(), engine_payload_path)

    print(f"Prepared data columns: {', '.join(signal_columns)}")
    print(f"Wrote config: {config_path}")
    print(f"Wrote engine capabilities: {engine_payload_path}")

    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(SRC_DIR)
        if not env.get("PYTHONPATH")
        else str(SRC_DIR) + os.pathsep + env["PYTHONPATH"]
    )

    if not args.skip_cli:
        if args.workflow == "variable-projection":
            command = [
                sys.executable,
                "-m",
                "odefit.cli",
                "fit-global-observables",
                "--config",
                str(config_path),
                "--variable-projection",
            ]
        else:
            command = [
                sys.executable,
                "-m",
                "odefit.cli",
                "fit",
                "--config",
                str(config_path),
            ]

        run_cli(command, env=env)

    if args.also_run_api_engine:
        output = fit_global_observables_from_config(config)
        payload = backend_output_payload(
            output,
            workflow="fit",
            max_rows=args.payload_rows,
        )
        payload_path = output_dir / "backend_payload.json"
        write_json(payload, payload_path)
        print(f"Wrote backend API payload: {payload_path}")


if __name__ == "__main__":
    main()

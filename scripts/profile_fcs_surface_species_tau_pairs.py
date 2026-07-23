from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, lsq_linear

from odefit.engines.registry import get_engine_bundle
from odefit.fitting.engine_helpers import engine_solve_to_dataframe
from odefit.fitting.fit_settings import FitSettings
from odefit.model.model_spec import build_model_spec


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def parse_tau_from_column(column: str, fallback_index: int) -> float:
    text = str(column)

    if text.startswith("G_tau_"):
        text = text.removeprefix("G_tau_")
    if text.endswith("_ms"):
        text = text.removesuffix("_ms")

    text = text.replace("p", ".")

    try:
        return float(text)
    except ValueError:
        pass

    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if matches:
        return float(matches[-1])

    return float(fallback_index)


def read_model_files(model_dir: Path) -> dict[str, str]:
    model_texts: dict[str, str] = {}

    for path in sorted(model_dir.glob("*.txt")):
        text = path.read_text().strip()
        if text:
            model_texts[path.stem] = text

    if not model_texts:
        raise ValueError(f"No non-empty .txt model files found in {model_dir}")

    return model_texts


def model_parameter_names(model) -> list[str]:
    parameters = getattr(model, "parameters", [])

    if isinstance(parameters, dict):
        return [str(name) for name in parameters]

    return [str(item) for item in parameters]


def model_species_names(model) -> list[str]:
    species = getattr(model, "species", [])

    if isinstance(species, dict):
        return [str(name) for name in species]

    return [str(item) for item in species]


def infer_species_index(species_name: str) -> int | None:
    match = re.search(r"(\d+)$", str(species_name))
    if match:
        return int(match.group(1))
    return None


def auto_split_species(
    species_names: list[str],
    *,
    fast_max_index: int | None,
) -> tuple[list[str], list[str]]:
    """
    Split model species into fast and slow FCS component groups.

    Species called F or starting with PF are always assigned to the slow
    component. For P-numbered species, fast_max_index means
    P1..Pfast_max_index are fast and larger P species are slow.
    """

    species_names = [str(species) for species in species_names]

    if not species_names:
        raise ValueError("Cannot split an empty species list.")

    protofibril_species = [
        species
        for species in species_names
        if species == "F" or species.startswith("PF")
    ]
    ordinary_species = [
        species for species in species_names if species not in protofibril_species
    ]

    if not ordinary_species:
        ordinary_species = species_names
        protofibril_species = []

    if fast_max_index is not None:
        fast = []
        slow = []

        for species in ordinary_species:
            index = infer_species_index(species)
            if index is not None and index <= fast_max_index:
                fast.append(species)
            else:
                slow.append(species)

        slow.extend(protofibril_species)

        if fast and slow:
            return fast, slow

    split = max(1, len(ordinary_species) // 3)
    fast = ordinary_species[:split]
    slow = ordinary_species[split:] + protofibril_species

    if not slow:
        slow = ordinary_species[-1:] + protofibril_species

    return fast, slow


def resolve_species_groups_for_model(
    *,
    species_names: list[str],
    requested_fast_species: list[str] | None,
    requested_slow_species: list[str] | None,
    fast_max_index: int | None,
) -> tuple[list[str], list[str]]:
    """
    Resolve fast/slow species groups for one model.

    Explicit user groups are intersected with the model's actual species list.
    If either resolved group is empty, fall back to automatic model-specific
    splitting.
    """

    available = [str(species) for species in species_names]
    available_set = set(available)

    auto_fast, auto_slow = auto_split_species(available, fast_max_index=fast_max_index)

    if requested_fast_species is None:
        fast = auto_fast
    else:
        fast = [
            str(species)
            for species in requested_fast_species
            if str(species) in available_set
        ]

    if requested_slow_species is None:
        slow = auto_slow
    else:
        slow = [
            str(species)
            for species in requested_slow_species
            if str(species) in available_set
        ]

    if not fast:
        fast = auto_fast
    if not slow:
        slow = auto_slow

    fast_set = set(fast)
    slow = [species for species in slow if species not in fast_set]

    if not slow:
        slow = [species for species in available if species not in fast_set]

    if not fast or not slow:
        raise ValueError(
            "Could not resolve non-empty fast/slow species groups. "
            f"available={available}, fast={fast}, slow={slow}"
        )

    return fast, slow


def load_raw_fcs_matrix(
    *,
    data_path: Path,
    time_column: str,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame, np.ndarray]:
    dataframe = pd.read_csv(data_path)

    if time_column not in dataframe.columns:
        raise ValueError(
            f"time_column={time_column!r} not found. "
            f"Available columns include {list(dataframe.columns[:10])}"
        )

    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
    signal_columns = [
        column for column in numeric_columns if str(column).startswith("G_tau_")
    ]

    if not signal_columns:
        signal_columns = [column for column in numeric_columns if column != time_column]

    if not signal_columns:
        raise ValueError("No numeric FCS signal columns found.")

    tau_values = np.asarray(
        [
            parse_tau_from_column(column, index)
            for index, column in enumerate(signal_columns)
        ],
        dtype=float,
    )

    order = np.argsort(tau_values)
    tau_values = tau_values[order]
    signal_columns = [signal_columns[index] for index in order]

    time_values = dataframe[time_column].to_numpy(dtype=float)
    observed_matrix = dataframe[signal_columns].to_numpy(dtype=float)

    return time_values, tau_values, signal_columns, dataframe, observed_matrix


def fcs_kernel(
    tau_ms: np.ndarray,
    *,
    tau_d_ms: float,
    triplet_fraction: float = 0.0,
    triplet_tau_ms: float = 1.0,
) -> np.ndarray:
    """
    FCS kernel:

        K(tau; tau_D, F, tau_m)
        = 1 / (1 + tau / tau_D) * (1 - F + F * exp(-tau / tau_m))

    The fitted surface is:

        G(time, tau) = 1 + A_fast*S_fast(time)*K_fast(tau)
                         + A_slow*S_slow(time)*K_slow(tau)
    """

    tau = np.asarray(tau_ms, dtype=float)
    tau_d_ms = max(float(tau_d_ms), 1e-300)
    triplet_tau_ms = max(float(triplet_tau_ms), 1e-300)
    triplet_fraction = float(np.clip(triplet_fraction, 0.0, 1.0))

    diffusion = 1.0 / (1.0 + tau / tau_d_ms)
    photophysics = (
        1.0 - triplet_fraction + triplet_fraction * np.exp(-tau / triplet_tau_ms)
    )

    return diffusion * photophysics


def build_initial_conditions(
    species_names: list[str],
    *,
    active_species: str | None,
) -> dict[str, float]:
    species_names = [str(species) for species in species_names]

    if not species_names:
        raise ValueError("No species names available.")

    active = str(active_species) if active_species is not None else species_names[0]

    return {species: 1.0 if species == active else 0.0 for species in species_names}


def simulate_all_species(
    *,
    engine_bundle,
    model,
    parameters: dict[str, float],
    initial_conditions: dict[str, float],
    time_values: np.ndarray,
    settings: FitSettings,
) -> pd.DataFrame:
    return engine_solve_to_dataframe(
        engine_bundle=engine_bundle,
        model=model,
        parameters=parameters,
        initial_conditions=initial_conditions,
        timepoints=time_values,
        settings=(
            settings.to_simulation_settings()
            if hasattr(settings, "to_simulation_settings")
            else None
        ),
    )


def aggregate_species_timecourse(
    simulation: pd.DataFrame, species_group: list[str]
) -> np.ndarray:
    missing = [
        species for species in species_group if species not in simulation.columns
    ]

    if missing:
        raise ValueError(
            f"Missing species in simulation output: {missing}. "
            f"Available columns: {list(simulation.columns)}"
        )

    return np.sum(simulation[species_group].to_numpy(dtype=float), axis=1)


def solve_linear_surface_coefficients(
    *,
    basis_matrices: list[np.ndarray],
    observed_matrix: np.ndarray,
    ridge: float,
    nonnegative_components: bool,
    fixed_baseline: float = 1.0,
    fit_mask_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Solve:

        observed ~= fixed_baseline + sum_j amplitude_j * basis_j

    The amplitudes are solved against observed - fixed_baseline. With the FCS
    equation here, fixed_baseline should normally be 1.0.
    """

    y = observed_matrix.reshape(-1)
    valid = np.isfinite(y)

    if fit_mask_matrix is not None:
        valid = valid & fit_mask_matrix.reshape(-1).astype(bool)

    if not np.any(valid):
        raise ValueError("No valid points available for linear FCS amplitude solve.")

    y_centered = y - float(fixed_baseline)

    columns = [np.asarray(basis, dtype=float).reshape(-1) for basis in basis_matrices]

    if not columns:
        raise ValueError("At least one basis matrix is required.")

    design = np.column_stack(columns)
    design_valid = design[valid]
    y_valid = y_centered[valid]

    if ridge > 0:
        penalty = np.sqrt(float(ridge)) * np.eye(design_valid.shape[1])
        design_aug = np.vstack([design_valid, penalty])
        y_aug = np.concatenate([y_valid, np.zeros(design_valid.shape[1])])
    else:
        design_aug = design_valid
        y_aug = y_valid

    if nonnegative_components:
        result = lsq_linear(design_aug, y_aug, bounds=(0.0, np.inf), method="trf")
        beta = result.x
    else:
        beta, *_ = np.linalg.lstsq(design_aug, y_aug, rcond=None)

    predicted = np.full_like(y, fill_value=float(fixed_baseline), dtype=float)

    for index, basis in enumerate(basis_matrices):
        predicted += beta[index] * basis.reshape(-1)

    predicted_matrix = predicted.reshape(observed_matrix.shape)
    residuals = observed_matrix - predicted_matrix
    rss = float(np.nansum(residuals.reshape(-1)[valid] ** 2))

    return beta, predicted_matrix, residuals, rss


def calculate_ic(
    *, rss: float, n_observations: int, n_parameters: int
) -> tuple[float, float, float]:
    n = max(int(n_observations), 1)
    k = max(int(n_parameters), 1)
    rmse = math.sqrt(max(rss, 0.0) / n)

    if rss <= 0:
        return rmse, -math.inf, -math.inf

    aic = n * math.log(rss / n) + 2 * k
    bic = n * math.log(rss / n) + k * math.log(n)

    return rmse, aic, bic


@dataclass
class SpeciesTauPairProfileRow:
    model_name: str
    success: bool
    tau_d_fast_ms: float
    tau_d_slow_ms: float
    alpha: float
    triplet_fraction: float
    triplet_tau_ms: float
    fast_species: str
    slow_species: str
    rss: float | None
    rmse: float | None
    aic: float | None
    bic: float | None
    nfev: int | None
    baseline: float | None
    amplitude_fast: float | None
    amplitude_slow: float | None
    fast_amplitude_fraction: float | None
    slow_amplitude_fraction: float | None
    n_ode_parameters: int | None
    n_total_parameters: int | None
    error_type: str | None
    error_message: str | None


def fit_model_for_species_tau_pair(
    *,
    model_name: str,
    model_text: str,
    time_values: np.ndarray,
    tau_values: np.ndarray,
    observed_matrix: np.ndarray,
    engine_name: str,
    tau_d_fast_ms: float,
    tau_d_slow_ms: float,
    alpha: float,
    triplet_fraction: float,
    triplet_tau_ms: float,
    fast_species: list[str] | None,
    slow_species: list[str] | None,
    fast_max_index: int | None,
    active_species: str | None,
    initial_guess: float,
    lower_bound: float,
    upper_bound: float,
    max_nfev: int,
    rtol: float,
    atol: float,
    ridge: float,
    nonnegative_components: bool,
    max_predicted_g: float | None,
    prediction_ceiling_penalty: float,
    save_surface_dir: Path | None,
) -> SpeciesTauPairProfileRow:
    model = build_model_spec(model_text, name=model_name)
    parameter_names = model_parameter_names(model)
    species_names = model_species_names(model)

    fast_species, slow_species = resolve_species_groups_for_model(
        species_names=species_names,
        requested_fast_species=fast_species,
        requested_slow_species=slow_species,
        fast_max_index=fast_max_index,
    )

    initial_conditions = build_initial_conditions(
        species_names, active_species=active_species
    )
    engine_bundle = get_engine_bundle(engine_name)

    settings = FitSettings(
        species_mapping={},
        use_normalized_data=False,
        method="trf",
        loss="linear",
        max_nfev=max_nfev,
        rtol=rtol,
        atol=atol,
    )

    finite_mask = np.isfinite(observed_matrix).reshape(-1)

    if not parameter_names:
        raise ValueError(f"No kinetic parameters found for model {model_name!r}")

    x0 = np.log10(np.full(len(parameter_names), initial_guess, dtype=float))
    lower = np.log10(np.full(len(parameter_names), lower_bound, dtype=float))
    upper = np.log10(np.full(len(parameter_names), upper_bound, dtype=float))

    f_fast = fcs_kernel(
        tau_values,
        tau_d_ms=tau_d_fast_ms,
        triplet_fraction=triplet_fraction,
        triplet_tau_ms=triplet_tau_ms,
    )
    f_slow = fcs_kernel(
        tau_values,
        tau_d_ms=tau_d_slow_ms,
        triplet_fraction=triplet_fraction,
        triplet_tau_ms=triplet_tau_ms,
    )

    state: dict[str, Any] = {
        "best_score": math.inf,
        "best_payload": None,
        "last_error_type": None,
        "last_error_message": None,
    }

    def residual(theta: np.ndarray) -> np.ndarray:
        try:
            parameters = {
                name: float(10.0 ** theta[index])
                for index, name in enumerate(parameter_names)
            }

            simulation = simulate_all_species(
                engine_bundle=engine_bundle,
                model=model,
                parameters=parameters,
                initial_conditions=initial_conditions,
                time_values=time_values,
                settings=settings,
            )

            fast_timecourse = aggregate_species_timecourse(simulation, fast_species)
            slow_timecourse = aggregate_species_timecourse(simulation, slow_species)

            basis_fast = fast_timecourse[:, None] * f_fast[None, :]
            basis_slow = slow_timecourse[:, None] * f_slow[None, :]

            beta, predicted, residuals, rss = solve_linear_surface_coefficients(
                basis_matrices=[basis_fast, basis_slow],
                observed_matrix=observed_matrix,
                ridge=ridge,
                nonnegative_components=nonnegative_components,
                fixed_baseline=1.0,
            )

            residuals_flat = residuals.reshape(-1)
            predicted_flat = predicted.reshape(-1)

            fit_residuals = residuals_flat[finite_mask]

            if max_predicted_g is not None:
                ceiling_excess = np.maximum(
                    predicted_flat[finite_mask] - float(max_predicted_g),
                    0.0,
                )
                fit_residuals = np.concatenate(
                    [
                        fit_residuals,
                        np.sqrt(float(prediction_ceiling_penalty)) * ceiling_excess,
                    ]
                )

            score = float(np.sum(fit_residuals**2))

            if score < state["best_score"]:
                state["best_score"] = score
                state["best_payload"] = {
                    "parameters": parameters,
                    "simulation": simulation,
                    "fast_timecourse": fast_timecourse,
                    "slow_timecourse": slow_timecourse,
                    "beta": beta,
                    "predicted": predicted,
                    "residuals": residuals,
                    "rss": rss,
                    "objective_score": score,
                    "predicted_max": float(np.nanmax(predicted)),
                }

            return fit_residuals

        except Exception as exc:
            state["last_error_type"] = type(exc).__name__
            state["last_error_message"] = str(exc)
            failure_length = int(finite_mask.sum())

            if max_predicted_g is not None:
                failure_length *= 2

            return np.full(failure_length, 1e6, dtype=float)

    result = least_squares(
        residual,
        x0=x0,
        bounds=(lower, upper),
        method="trf",
        max_nfev=max_nfev,
        xtol=rtol,
        ftol=rtol,
        gtol=rtol,
    )

    residual(result.x)

    if state["best_payload"] is None:
        last_error_type = state.get("last_error_type")
        last_error_message = state.get("last_error_message")

        if last_error_type or last_error_message:
            raise RuntimeError(
                "No valid objective evaluation. "
                f"Last objective error was {last_error_type}: {last_error_message}"
            )

        raise RuntimeError("No valid objective evaluation.")

    payload = state["best_payload"]
    beta = payload["beta"]

    if len(beta) < 2:
        raise RuntimeError(
            f"Expected at least two linear amplitudes, got beta={beta!r}"
        )

    n_observations = int(finite_mask.sum())
    n_ode_parameters = len(parameter_names)
    n_total_parameters = n_ode_parameters + 2 + 2
    # ODE parameters + two profiled tau_D values + triplet fraction/tau_m.
    # Linear amplitudes are solved by variable projection.

    rmse, aic, bic = calculate_ic(
        rss=payload["rss"],
        n_observations=n_observations,
        n_parameters=n_total_parameters,
    )

    amp_fast = float(beta[0])
    amp_slow = float(beta[1])
    amp_sum = max(amp_fast, 0.0) + max(amp_slow, 0.0)

    fast_fraction = max(amp_fast, 0.0) / amp_sum if amp_sum > 0 else np.nan
    slow_fraction = max(amp_slow, 0.0) / amp_sum if amp_sum > 0 else np.nan

    if save_surface_dir is not None:
        save_surface_dir.mkdir(parents=True, exist_ok=True)

        predicted_columns = [f"G_tau_{tau:.8g}_ms" for tau in tau_values]

        predicted_dataframe = pd.DataFrame(
            payload["predicted"], columns=predicted_columns
        )
        predicted_dataframe.insert(0, "time_min", time_values)
        predicted_dataframe.to_csv(
            save_surface_dir / "species_surface_predicted.csv", index=False
        )

        residuals_dataframe = pd.DataFrame(
            payload["residuals"], columns=predicted_columns
        )
        residuals_dataframe.insert(0, "time_min", time_values)
        residuals_dataframe.to_csv(
            save_surface_dir / "species_surface_residuals.csv", index=False
        )

        timecourses = pd.DataFrame(
            {
                "time_min": time_values,
                "fast_species_timecourse": payload["fast_timecourse"],
                "slow_species_timecourse": payload["slow_timecourse"],
            }
        )
        timecourses.to_csv(
            save_surface_dir / "species_component_timecourses.csv", index=False
        )

        payload["simulation"].to_csv(
            save_surface_dir / "ode_simulation.csv", index=False
        )

        summary = {
            "model_name": model_name,
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "tau_d_fast_ms": float(tau_d_fast_ms),
            "tau_d_slow_ms": float(tau_d_slow_ms),
            "alpha": float(alpha),
            "triplet_fraction": float(triplet_fraction),
            "triplet_tau_ms": float(triplet_tau_ms),
            "fast_species": fast_species,
            "slow_species": slow_species,
            "rss": float(payload["rss"]),
            "objective_score": float(payload["objective_score"]),
            "predicted_max": float(payload["predicted_max"]),
            "rmse": float(rmse),
            "aic": float(aic),
            "bic": float(bic),
            "baseline": 1.0,
            "amplitude_fast": amp_fast,
            "amplitude_slow": amp_slow,
            "fast_amplitude_fraction": float(fast_fraction),
            "slow_amplitude_fraction": float(slow_fraction),
            "fitted_parameters": payload["parameters"],
        }

        with (save_surface_dir / "species_surface_fit_summary.json").open(
            "w"
        ) as handle:
            json.dump(summary, handle, indent=2)

    return SpeciesTauPairProfileRow(
        model_name=model_name,
        success=bool(result.success or state["best_payload"] is not None),
        tau_d_fast_ms=float(tau_d_fast_ms),
        tau_d_slow_ms=float(tau_d_slow_ms),
        alpha=float(alpha),
        triplet_fraction=float(triplet_fraction),
        triplet_tau_ms=float(triplet_tau_ms),
        fast_species=" ".join(fast_species),
        slow_species=" ".join(slow_species),
        rss=float(payload["rss"]),
        rmse=float(rmse),
        aic=float(aic),
        bic=float(bic),
        nfev=int(result.nfev),
        baseline=1.0,
        amplitude_fast=amp_fast,
        amplitude_slow=amp_slow,
        fast_amplitude_fraction=float(fast_fraction),
        slow_amplitude_fraction=float(slow_fraction),
        n_ode_parameters=int(n_ode_parameters),
        n_total_parameters=int(n_total_parameters),
        error_type=None,
        error_message=None,
    )


def compute_weights(dataframe: pd.DataFrame, criterion: str) -> pd.DataFrame:
    output = dataframe.copy()

    if criterion not in output.columns:
        return output

    values = pd.to_numeric(output[criterion], errors="coerce")
    valid = values.notna() & output["success"].astype(bool)

    output[f"delta_{criterion}"] = np.nan
    output[f"{criterion}_weight"] = np.nan
    output[f"{criterion}_weight_percent"] = np.nan

    if valid.any():
        delta = values[valid] - values[valid].min()
        raw = np.exp(-0.5 * delta.to_numpy(dtype=float))
        weights = raw / raw.sum()

        output.loc[valid, f"delta_{criterion}"] = delta
        output.loc[valid, f"{criterion}_weight"] = weights
        output.loc[valid, f"{criterion}_weight_percent"] = 100.0 * weights

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Profile two-tau_D FCS surface fits where fast and slow tau components "
            "use different ODE species groups."
        )
    )

    parser.add_argument("--data", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", default="fcs_ode/profile_species_tau_pairs")
    parser.add_argument("--engine-name", default="numba_projection")
    parser.add_argument("--time-column", default="time_min")

    parser.add_argument("--fast-species", nargs="+", default=None)
    parser.add_argument("--slow-species", nargs="+", default=None)
    parser.add_argument(
        "--fast-max-index",
        type=int,
        default=3,
        help="Auto split P1..Pfast_max_index as fast if explicit groups are not provided.",
    )
    parser.add_argument("--initial-active-species", default=None)

    parser.add_argument("--tau-min-ms", type=float, default=1e-3)
    parser.add_argument("--tau-max-ms", type=float, default=1e5)
    parser.add_argument("--n-tau-grid", type=int, default=8)
    parser.add_argument("--min-ratio", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=1.0)

    parser.add_argument(
        "--triplet-fraction",
        type=float,
        default=0.0,
        help="F in the FCS photophysics term: 1 - F + F * exp(-tau / tau_m).",
    )
    parser.add_argument(
        "--triplet-tau-ms",
        type=float,
        default=1.0,
        help="tau_m in ms for the FCS photophysics term.",
    )

    parser.add_argument("--max-predicted-g", type=float, default=None)
    parser.add_argument("--prediction-ceiling-penalty", type=float, default=100.0)

    parser.add_argument("--initial-guess", type=float, default=0.1)
    parser.add_argument("--lower-bound", type=float, default=1e-8)
    parser.add_argument("--upper-bound", type=float, default=100.0)
    parser.add_argument("--max-nfev", type=int, default=50)
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--allow-negative-components", action="store_true")

    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--save-best-surfaces",
        action="store_true",
        help="Save predicted/residual surfaces for the best tau-pair of each model.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    time_values, tau_values, signal_columns, raw_dataframe, observed_matrix = (
        load_raw_fcs_matrix(
            data_path=Path(args.data),
            time_column=args.time_column,
        )
    )

    model_texts = read_model_files(Path(args.model_dir))

    if args.model_name:
        model_texts = {
            name: text for name, text in model_texts.items() if name == args.model_name
        }
        if not model_texts:
            raise ValueError(f"Model {args.model_name!r} not found.")

    preview_rows = []

    for model_name, model_text in model_texts.items():
        model = build_model_spec(model_text, name=model_name)
        species_names = model_species_names(model)
        parameter_names = model_parameter_names(model)

        resolved_fast, resolved_slow = resolve_species_groups_for_model(
            species_names=species_names,
            requested_fast_species=args.fast_species,
            requested_slow_species=args.slow_species,
            fast_max_index=args.fast_max_index,
        )

        preview_rows.append(
            {
                "model_name": model_name,
                "n_species": len(species_names),
                "n_parameters": len(parameter_names),
                "species": " ".join(species_names),
                "fast_species": " ".join(resolved_fast),
                "slow_species": " ".join(resolved_slow),
            }
        )

    pd.DataFrame(preview_rows).to_csv(
        output_dir / "species_group_preview.csv", index=False
    )

    tau_grid = np.logspace(
        np.log10(args.tau_min_ms), np.log10(args.tau_max_ms), args.n_tau_grid
    )

    pairs = []
    for fast, slow in itertools.product(tau_grid, tau_grid):
        if slow <= fast:
            continue
        if slow / fast < args.min_ratio:
            continue
        pairs.append((float(fast), float(slow)))

    run_config = vars(args).copy()
    run_config["tau_grid"] = tau_grid.tolist()
    run_config["n_tau_pairs"] = len(pairs)
    run_config["n_timepoints"] = int(len(time_values))
    run_config["n_tau_points"] = int(len(tau_values))

    with (output_dir / "species_tau_pair_profile_config.json").open("w") as handle:
        json.dump(run_config, handle, indent=2)

    pd.DataFrame({"column": signal_columns, "tau_ms": tau_values}).to_csv(
        output_dir / "surface_tau_column_map.csv",
        index=False,
    )

    print("\nSpecies-group two-tau_D FCS surface profile")
    print("===========================================")
    print(f"Data: {args.data}")
    print(f"Models: {list(model_texts)}")
    print(f"Surface shape: {observed_matrix.shape}")
    print(f"Tau grid: {tau_grid}")
    print(f"Tau pairs: {len(pairs)}")
    print(f"Fast species: {args.fast_species or f'auto P1..P{args.fast_max_index}'}")
    print(f"Slow species: {args.slow_species or 'auto remaining species / F / PF'}")
    print(f"Triplet fraction: {args.triplet_fraction}")
    print(f"Triplet tau ms: {args.triplet_tau_ms}")
    print(f"Max predicted G: {args.max_predicted_g}")

    rows: list[SpeciesTauPairProfileRow] = []
    best_rows_by_model: dict[str, SpeciesTauPairProfileRow] = {}

    for model_name, model_text in model_texts.items():
        print(f"\nModel {model_name}: {len(pairs)} tau pairs", flush=True)
        model_rows = []

        for index, (tau_fast, tau_slow) in enumerate(pairs, start=1):
            try:
                row = fit_model_for_species_tau_pair(
                    model_name=model_name,
                    model_text=model_text,
                    time_values=time_values,
                    tau_values=tau_values,
                    observed_matrix=observed_matrix,
                    engine_name=args.engine_name,
                    tau_d_fast_ms=tau_fast,
                    tau_d_slow_ms=tau_slow,
                    alpha=args.alpha,
                    triplet_fraction=args.triplet_fraction,
                    triplet_tau_ms=args.triplet_tau_ms,
                    fast_species=args.fast_species,
                    slow_species=args.slow_species,
                    fast_max_index=args.fast_max_index,
                    active_species=args.initial_active_species,
                    initial_guess=args.initial_guess,
                    lower_bound=args.lower_bound,
                    upper_bound=args.upper_bound,
                    max_nfev=args.max_nfev,
                    rtol=args.rtol,
                    atol=args.atol,
                    ridge=args.ridge,
                    nonnegative_components=not args.allow_negative_components,
                    max_predicted_g=args.max_predicted_g,
                    prediction_ceiling_penalty=args.prediction_ceiling_penalty,
                    save_surface_dir=None,
                )
            except Exception as exc:
                row = SpeciesTauPairProfileRow(
                    model_name=model_name,
                    success=False,
                    tau_d_fast_ms=float(tau_fast),
                    tau_d_slow_ms=float(tau_slow),
                    alpha=float(args.alpha),
                    triplet_fraction=float(args.triplet_fraction),
                    triplet_tau_ms=float(args.triplet_tau_ms),
                    fast_species=" ".join(args.fast_species or []),
                    slow_species=" ".join(args.slow_species or []),
                    rss=None,
                    rmse=None,
                    aic=None,
                    bic=None,
                    nfev=None,
                    baseline=None,
                    amplitude_fast=None,
                    amplitude_slow=None,
                    fast_amplitude_fraction=None,
                    slow_amplitude_fraction=None,
                    n_ode_parameters=None,
                    n_total_parameters=None,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )

            rows.append(row)
            model_rows.append(row)

            if index == 1 or index == len(pairs) or index % 10 == 0:
                successful = [
                    item for item in model_rows if item.success and item.bic is not None
                ]
                best_bic = min((item.bic for item in successful), default=None)
                print(
                    f"{model_name}: {index}/{len(pairs)} "
                    f"tau_fast={tau_fast:.4g} tau_slow={tau_slow:.4g} "
                    f"best_bic={best_bic}",
                    flush=True,
                )

        successful_rows = [
            row for row in model_rows if row.success and row.bic is not None
        ]
        if successful_rows:
            best_rows_by_model[model_name] = min(
                successful_rows, key=lambda row: float(row.bic)
            )

    dataframe = pd.DataFrame([asdict(row) for row in rows])
    dataframe = compute_weights(dataframe, "bic")
    dataframe = compute_weights(dataframe, "aic")

    if "bic" in dataframe.columns:
        dataframe = dataframe.sort_values("bic", na_position="last")

    profile_path = output_dir / "species_tau_pair_profile.csv"
    dataframe.to_csv(profile_path, index=False)

    best_dataframe = (
        pd.DataFrame([asdict(row) for row in best_rows_by_model.values()])
        if best_rows_by_model
        else pd.DataFrame()
    )

    if not best_dataframe.empty:
        best_dataframe = compute_weights(best_dataframe, "bic")
        best_dataframe = compute_weights(best_dataframe, "aic")
        best_dataframe = best_dataframe.sort_values("bic", na_position="last")

    best_path = output_dir / "species_tau_pair_profile_best_by_model.csv"
    best_dataframe.to_csv(best_path, index=False)

    if args.save_best_surfaces:
        for model_name, best in best_rows_by_model.items():
            print(f"Saving best surface for {model_name}", flush=True)
            fit_model_for_species_tau_pair(
                model_name=model_name,
                model_text=model_texts[model_name],
                time_values=time_values,
                tau_values=tau_values,
                observed_matrix=observed_matrix,
                engine_name=args.engine_name,
                tau_d_fast_ms=best.tau_d_fast_ms,
                tau_d_slow_ms=best.tau_d_slow_ms,
                alpha=args.alpha,
                triplet_fraction=args.triplet_fraction,
                triplet_tau_ms=args.triplet_tau_ms,
                fast_species=args.fast_species,
                slow_species=args.slow_species,
                fast_max_index=args.fast_max_index,
                active_species=args.initial_active_species,
                initial_guess=args.initial_guess,
                lower_bound=args.lower_bound,
                upper_bound=args.upper_bound,
                max_nfev=args.max_nfev,
                rtol=args.rtol,
                atol=args.atol,
                ridge=args.ridge,
                nonnegative_components=not args.allow_negative_components,
                max_predicted_g=args.max_predicted_g,
                prediction_ceiling_penalty=args.prediction_ceiling_penalty,
                save_surface_dir=output_dir / "models" / safe_name(model_name),
            )

    print("\nSpecies tau-pair profile complete")
    print("=================================")
    print(f"Wrote profile: {profile_path}")
    print(f"Wrote best table: {best_path}")

    if not best_dataframe.empty:
        display_columns = [
            "model_name",
            "tau_d_fast_ms",
            "tau_d_slow_ms",
            "triplet_fraction",
            "triplet_tau_ms",
            "rss",
            "aic",
            "bic",
            "bic_weight_percent",
            "amplitude_fast",
            "amplitude_slow",
            "fast_amplitude_fraction",
            "slow_amplitude_fraction",
            "nfev",
        ]
        display_columns = [
            column for column in display_columns if column in best_dataframe.columns
        ]
        print(best_dataframe[display_columns].to_string(index=False))
    else:
        print("No successful fits.")


if __name__ == "__main__":
    main()

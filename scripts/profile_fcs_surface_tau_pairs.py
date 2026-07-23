from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

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
    model_texts = {}

    for path in sorted(model_dir.glob("*.txt")):
        text = path.read_text().strip()

        if text:
            model_texts[path.stem] = text

    if not model_texts:
        raise ValueError(f"No .txt model files found in {model_dir}")

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


def load_raw_fcs_matrix(
    *,
    data_path: Path,
    time_column: str,
):
    dataframe = pd.read_csv(data_path)

    if time_column not in dataframe.columns:
        raise ValueError(f"Missing time column: {time_column}")

    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()

    signal_columns = [
        column
        for column in numeric_columns
        if str(column).startswith("G_tau_")
    ]

    if not signal_columns:
        signal_columns = [
            column
            for column in numeric_columns
            if column != time_column
        ]

    tau_values = np.array(
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

    return time_values, tau_values, signal_columns, observed_matrix


def fcs_kernel(
    tau_ms: np.ndarray,
    *,
    tau_d_ms: float,
    alpha: float,
) -> np.ndarray:
    return 1.0 / np.power(
        1.0 + tau_ms / max(float(tau_d_ms), 1e-30),
        max(float(alpha), 1e-12),
    )


def build_initial_conditions(
    species_names: list[str],
    *,
    active_species: str | None,
) -> dict[str, float]:
    active = active_species or species_names[0]

    return {
        species: 1.0 if species == active else 0.0
        for species in species_names
    }


def simulate_species(
    *,
    engine_bundle,
    model,
    parameters: dict[str, float],
    initial_conditions: dict[str, float],
    time_values: np.ndarray,
    observed_species: str,
    settings: FitSettings,
) -> np.ndarray:
    simulation = engine_solve_to_dataframe(
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

    if observed_species not in simulation.columns:
        raise ValueError(
            f"Observed species {observed_species!r} not found in simulation columns "
            f"{list(simulation.columns)}"
        )

    return simulation[observed_species].to_numpy(dtype=float)


def solve_linear_surface_coefficients(
    *,
    basis_matrices: list[np.ndarray],
    observed_matrix: np.ndarray,
    ridge: float,
    nonnegative_components: bool,
):
    """
    Solve:

        observed ~= baseline + sum_j amplitude_j * basis_j

    If nonnegative_components is true, amplitudes are clipped after least-squares.
    This is a pragmatic stabilizer, not a full NNLS solver.
    """

    y = observed_matrix.reshape(-1)
    valid = np.isfinite(y)

    columns = [np.ones_like(y)]

    for basis in basis_matrices:
        columns.append(basis.reshape(-1))

    design = np.column_stack(columns)
    design_valid = design[valid]
    y_valid = y[valid]

    if ridge > 0:
        lhs = design_valid.T @ design_valid
        rhs = design_valid.T @ y_valid

        penalty = ridge * np.eye(lhs.shape[0])
        penalty[0, 0] = 0.0

        beta = np.linalg.solve(lhs + penalty, rhs)
    else:
        beta, *_ = np.linalg.lstsq(design_valid, y_valid, rcond=None)

    if nonnegative_components:
        beta[1:] = np.maximum(beta[1:], 0.0)

    predicted = np.zeros_like(y, dtype=float) + beta[0]

    for index, basis in enumerate(basis_matrices, start=1):
        predicted += beta[index] * basis.reshape(-1)

    predicted_matrix = predicted.reshape(observed_matrix.shape)
    residuals = observed_matrix - predicted_matrix
    rss = float(np.nansum(residuals[np.isfinite(residuals)] ** 2))

    return beta, predicted_matrix, residuals, rss


def calculate_ic(
    *,
    rss: float,
    n_observations: int,
    n_parameters: int,
):
    n = max(int(n_observations), 1)
    k = max(int(n_parameters), 1)

    rmse = math.sqrt(max(rss, 0.0) / n)

    if rss <= 0:
        return rmse, -math.inf, -math.inf

    aic = n * math.log(rss / n) + 2 * k
    bic = n * math.log(rss / n) + k * math.log(n)

    return rmse, aic, bic


@dataclass
class TauPairProfileRow:
    model_name: str
    tau_d_fast_ms: float
    tau_d_slow_ms: float
    alpha: float
    success: bool
    rss: float | None
    rmse: float | None
    aic: float | None
    bic: float | None
    nfev: int | None
    baseline: float | None
    amplitude_fast: float | None
    amplitude_slow: float | None
    error_type: str | None
    error_message: str | None


def fit_model_for_tau_pair(
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
    observed_species: str | None,
    active_species: str | None,
    initial_guess: float,
    lower_bound: float,
    upper_bound: float,
    max_nfev: int,
    rtol: float,
    atol: float,
    ridge: float,
    nonnegative_components: bool,
) -> TauPairProfileRow:
    model = build_model_spec(
        model_text,
        name=model_name,
    )

    parameter_names = model_parameter_names(model)
    species_names = model_species_names(model)

    selected_species = observed_species or species_names[0]

    initial_conditions = build_initial_conditions(
        species_names,
        active_species=active_species,
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

    finite_mask = np.isfinite(observed_matrix)

    x0 = np.log10(np.full(len(parameter_names), initial_guess, dtype=float))
    lower = np.log10(np.full(len(parameter_names), lower_bound, dtype=float))
    upper = np.log10(np.full(len(parameter_names), upper_bound, dtype=float))

    f_fast = fcs_kernel(
        tau_values,
        tau_d_ms=tau_d_fast_ms,
        alpha=alpha,
    )

    f_slow = fcs_kernel(
        tau_values,
        tau_d_ms=tau_d_slow_ms,
        alpha=alpha,
    )

    state = {
        "best_rss": math.inf,
        "best_payload": None,
    }

    def residual(theta):
        try:
            parameters = {
                name: float(10.0 ** theta[index])
                for index, name in enumerate(parameter_names)
            }

            species_values = simulate_species(
                engine_bundle=engine_bundle,
                model=model,
                parameters=parameters,
                initial_conditions=initial_conditions,
                time_values=time_values,
                observed_species=selected_species,
                settings=settings,
            )

            basis_fast = species_values[:, None] * f_fast[None, :]
            basis_slow = species_values[:, None] * f_slow[None, :]

            beta, predicted, residuals, rss = solve_linear_surface_coefficients(
                basis_matrices=[basis_fast, basis_slow],
                observed_matrix=observed_matrix,
                ridge=ridge,
                nonnegative_components=nonnegative_components,
            )

            if rss < state["best_rss"]:
                state["best_rss"] = rss
                state["best_payload"] = {
                    "parameters": parameters,
                    "beta": beta,
                    "predicted": predicted,
                    "residuals": residuals,
                    "rss": rss,
                }

            return residuals[finite_mask]

        except Exception:
            return np.full(int(finite_mask.sum()), 1e6, dtype=float)

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
        raise RuntimeError("No valid objective evaluation.")

    payload = state["best_payload"]
    beta = payload["beta"]

    n_observations = int(finite_mask.sum())
    n_parameters = len(parameter_names) + 2 + 1 + 3
    # ODE parameters + two fixed/profiled tau_Ds + alpha + baseline/two amplitudes.

    rmse, aic, bic = calculate_ic(
        rss=payload["rss"],
        n_observations=n_observations,
        n_parameters=n_parameters,
    )

    return TauPairProfileRow(
        model_name=model_name,
        tau_d_fast_ms=float(tau_d_fast_ms),
        tau_d_slow_ms=float(tau_d_slow_ms),
        alpha=float(alpha),
        success=bool(result.success),
        rss=float(payload["rss"]),
        rmse=float(rmse),
        aic=float(aic),
        bic=float(bic),
        nfev=int(result.nfev),
        baseline=float(beta[0]),
        amplitude_fast=float(beta[1]),
        amplitude_slow=float(beta[2]),
        error_type=None,
        error_message=None,
    )


def profile_model_tau_pairs(
    *,
    model_name: str,
    model_text: str,
    time_values: np.ndarray,
    tau_values: np.ndarray,
    observed_matrix: np.ndarray,
    engine_name: str,
    tau_grid: np.ndarray,
    min_ratio: float,
    alpha: float,
    observed_species: str | None,
    active_species: str | None,
    initial_guess: float,
    lower_bound: float,
    upper_bound: float,
    max_nfev: int,
    rtol: float,
    atol: float,
    ridge: float,
    nonnegative_components: bool,
) -> list[TauPairProfileRow]:
    rows = []

    pairs = []

    for fast, slow in itertools.product(tau_grid, tau_grid):
        if slow <= fast:
            continue

        if slow / fast < min_ratio:
            continue

        pairs.append((float(fast), float(slow)))

    total = len(pairs)

    print(f"\nModel {model_name}: profiling {total} tau_D pairs", flush=True)

    for index, (fast, slow) in enumerate(pairs, start=1):
        try:
            row = fit_model_for_tau_pair(
                model_name=model_name,
                model_text=model_text,
                time_values=time_values,
                tau_values=tau_values,
                observed_matrix=observed_matrix,
                engine_name=engine_name,
                tau_d_fast_ms=fast,
                tau_d_slow_ms=slow,
                alpha=alpha,
                observed_species=observed_species,
                active_species=active_species,
                initial_guess=initial_guess,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                max_nfev=max_nfev,
                rtol=rtol,
                atol=atol,
                ridge=ridge,
                nonnegative_components=nonnegative_components,
            )

        except Exception as exc:
            row = TauPairProfileRow(
                model_name=model_name,
                tau_d_fast_ms=fast,
                tau_d_slow_ms=slow,
                alpha=alpha,
                success=False,
                rss=None,
                rmse=None,
                aic=None,
                bic=None,
                nfev=None,
                baseline=None,
                amplitude_fast=None,
                amplitude_slow=None,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

        rows.append(row)

        if index == 1 or index == total or index % 10 == 0:
            best = min(
                [
                    item.rss
                    for item in rows
                    if item.rss is not None
                ],
                default=None,
            )
            print(
                f"{model_name}: {index}/{total} "
                f"tau_fast={fast:.4g} tau_slow={slow:.4g} "
                f"best_rss={best}",
                flush=True,
            )

    return rows


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
        description="Profile two-tau_D FCS surface fits over fixed tau_D pairs."
    )

    parser.add_argument("--data", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", default="fcs_ode/profile_fcs_tau_pairs")
    parser.add_argument("--engine-name", default="numba_projection")
    parser.add_argument("--time-column", default="time_min")
    parser.add_argument("--observed-species", default=None)
    parser.add_argument("--initial-active-species", default=None)

    parser.add_argument("--tau-min-ms", type=float, default=1e-3)
    parser.add_argument("--tau-max-ms", type=float, default=1e5)
    parser.add_argument("--n-tau-grid", type=int, default=8)
    parser.add_argument("--min-ratio", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=1.0)

    parser.add_argument("--initial-guess", type=float, default=0.1)
    parser.add_argument("--lower-bound", type=float, default=1e-8)
    parser.add_argument("--upper-bound", type=float, default=100.0)
    parser.add_argument("--max-nfev", type=int, default=50)
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--allow-negative-components", action="store_true")
    parser.add_argument("--model-name", default=None)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    time_values, tau_values, signal_columns, observed_matrix = load_raw_fcs_matrix(
        data_path=Path(args.data),
        time_column=args.time_column,
    )

    model_texts = read_model_files(Path(args.model_dir))

    if args.model_name:
        model_texts = {
            name: text
            for name, text in model_texts.items()
            if name == args.model_name
        }

        if not model_texts:
            raise ValueError(f"Model {args.model_name!r} not found.")

    tau_grid = np.logspace(
        np.log10(args.tau_min_ms),
        np.log10(args.tau_max_ms),
        args.n_tau_grid,
    )

    print("\nTwo-tau_D FCS surface profile")
    print("=============================")
    print(f"Data: {args.data}")
    print(f"Models: {list(model_texts)}")
    print(f"Tau grid: {tau_grid}")
    print(f"Min slow/fast ratio: {args.min_ratio}")
    print(f"Surface: {observed_matrix.shape}")

    all_rows = []

    for model_name, model_text in model_texts.items():
        rows = profile_model_tau_pairs(
            model_name=model_name,
            model_text=model_text,
            time_values=time_values,
            tau_values=tau_values,
            observed_matrix=observed_matrix,
            engine_name=args.engine_name,
            tau_grid=tau_grid,
            min_ratio=args.min_ratio,
            alpha=args.alpha,
            observed_species=args.observed_species,
            active_species=args.initial_active_species,
            initial_guess=args.initial_guess,
            lower_bound=args.lower_bound,
            upper_bound=args.upper_bound,
            max_nfev=args.max_nfev,
            rtol=args.rtol,
            atol=args.atol,
            ridge=args.ridge,
            nonnegative_components=not args.allow_negative_components,
        )

        all_rows.extend(rows)

    dataframe = pd.DataFrame([asdict(row) for row in all_rows])

    dataframe = compute_weights(dataframe, "bic")
    dataframe = compute_weights(dataframe, "aic")

    if "bic" in dataframe.columns:
        dataframe = dataframe.sort_values("bic", na_position="last")

    csv_path = output_dir / "fcs_tau_pair_profile.csv"
    json_path = output_dir / "fcs_tau_pair_profile.json"
    best_path = output_dir / "fcs_tau_pair_profile_best_by_model.csv"

    dataframe.to_csv(csv_path, index=False)

    with json_path.open("w") as handle:
        json.dump(dataframe.to_dict(orient="records"), handle, indent=2)

    successful = dataframe[dataframe["success"].astype(bool)].copy()

    if not successful.empty:
        best_by_model = (
            successful.sort_values("bic")
            .groupby("model_name", as_index=False)
            .head(1)
            .sort_values("bic")
        )
    else:
        best_by_model = pd.DataFrame()

    best_by_model.to_csv(best_path, index=False)

    print("\nTau-pair profile complete")
    print("=========================")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {best_path}")

    if not best_by_model.empty:
        cols = [
            "model_name",
            "tau_d_fast_ms",
            "tau_d_slow_ms",
            "rss",
            "aic",
            "bic",
            "bic_weight_percent",
            "amplitude_fast",
            "amplitude_slow",
            "nfev",
        ]
        cols = [column for column in cols if column in best_by_model.columns]
        print(best_by_model[cols].to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear


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


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def load_fcs_profiles(
    *,
    data_path: Path,
    time_column: str,
) -> tuple[pd.DataFrame, np.ndarray, list[str], np.ndarray]:
    dataframe = pd.read_csv(data_path)

    if time_column not in dataframe.columns:
        raise ValueError(
            f"Time column {time_column!r} not found. "
            f"Available columns include {list(dataframe.columns[:10])}"
        )

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

    if not signal_columns:
        raise ValueError("No numeric FCS G(tau) columns found.")

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

    matrix = dataframe[signal_columns].to_numpy(dtype=float)

    return dataframe, tau_values, signal_columns, matrix


def fcs_kernel(
    tau_ms: np.ndarray,
    *,
    tau_d_ms: float,
    alpha: float,
) -> np.ndarray:
    tau_d_ms = max(float(tau_d_ms), 1e-300)
    alpha = max(float(alpha), 1e-12)

    return 1.0 / np.power(
        1.0 + tau_ms / tau_d_ms,
        alpha,
    )


def solve_linear_profile(
    *,
    basis_columns: list[np.ndarray],
    observed: np.ndarray,
    nonnegative_amplitudes: bool,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Solve:

        observed ~= baseline + sum_j amplitude_j * basis_j

    baseline is unconstrained.
    amplitudes are optionally constrained >= 0.
    """

    y = np.asarray(observed, dtype=float)
    valid = np.isfinite(y)

    if valid.sum() < 5:
        raise ValueError("Too few finite profile points.")

    design_columns = [np.ones_like(y)]

    for basis in basis_columns:
        design_columns.append(np.asarray(basis, dtype=float))

    design = np.column_stack(design_columns)

    x = design[valid]
    y_valid = y[valid]

    if ridge > 0:
        # Ridge by row augmentation. Do not penalize baseline.
        penalty = math.sqrt(ridge) * np.eye(x.shape[1])
        penalty[0, 0] = 0.0
        x_aug = np.vstack([x, penalty])
        y_aug = np.concatenate([y_valid, np.zeros(x.shape[1])])
    else:
        x_aug = x
        y_aug = y_valid

    if nonnegative_amplitudes:
        lower = np.full(x_aug.shape[1], -np.inf)
        upper = np.full(x_aug.shape[1], np.inf)

        lower[1:] = 0.0

        result = lsq_linear(
            x_aug,
            y_aug,
            bounds=(lower, upper),
            method="trf",
        )

        coefficients = result.x
    else:
        coefficients, *_ = np.linalg.lstsq(
            x_aug,
            y_aug,
            rcond=None,
        )

    predicted = design @ coefficients
    residuals = y - predicted

    rss = float(np.nansum(residuals[valid] ** 2))

    return coefficients, predicted, residuals, rss


def calculate_ic(
    *,
    rss: float,
    n_observations: int,
    n_parameters: int,
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
class ProfileFit:
    model_type: str
    success: bool
    rss: float
    rmse: float
    aic: float
    bic: float
    baseline: float
    amplitude_fast: float | None
    amplitude_slow: float | None
    tau_d_fast_ms: float | None
    tau_d_slow_ms: float | None
    alpha: float
    predicted: list[float]
    residuals: list[float]


@dataclass
class TimepointDecision:
    row_index: int
    time_value: float
    selected_model: str
    two_tau_supported: bool
    reason: str

    single_rss: float
    single_bic: float
    single_aic: float
    single_tau_d_ms: float
    single_alpha: float

    two_rss: float | None
    two_bic: float | None
    two_aic: float | None
    two_tau_d_fast_ms: float | None
    two_tau_d_slow_ms: float | None
    two_alpha: float | None
    two_amplitude_fast: float | None
    two_amplitude_slow: float | None

    delta_bic_single_minus_two: float | None
    delta_aic_single_minus_two: float | None
    two_bic_weight_percent: float | None
    two_aic_weight_percent: float | None
    rss_improvement_fraction: float | None
    slow_fast_ratio: float | None
    fast_amplitude_fraction: float | None
    slow_amplitude_fraction: float | None


def fit_single_tau_profile(
    *,
    observed: np.ndarray,
    tau_values: np.ndarray,
    tau_grid: np.ndarray,
    alpha_grid: np.ndarray,
    nonnegative_amplitudes: bool,
    ridge: float,
    alpha_count_as_parameter: bool,
) -> ProfileFit:
    best = None

    finite_count = int(np.isfinite(observed).sum())

    for tau_d_ms, alpha in itertools.product(tau_grid, alpha_grid):
        kernel = fcs_kernel(
            tau_values,
            tau_d_ms=float(tau_d_ms),
            alpha=float(alpha),
        )

        coefficients, predicted, residuals, rss = solve_linear_profile(
            basis_columns=[kernel],
            observed=observed,
            nonnegative_amplitudes=nonnegative_amplitudes,
            ridge=ridge,
        )

        n_parameters = 3 + int(alpha_count_as_parameter)
        # baseline, amplitude, tau_D, optional alpha.

        rmse, aic, bic = calculate_ic(
            rss=rss,
            n_observations=finite_count,
            n_parameters=n_parameters,
        )

        fit = ProfileFit(
            model_type="single_tau",
            success=True,
            rss=float(rss),
            rmse=float(rmse),
            aic=float(aic),
            bic=float(bic),
            baseline=float(coefficients[0]),
            amplitude_fast=float(coefficients[1]),
            amplitude_slow=None,
            tau_d_fast_ms=float(tau_d_ms),
            tau_d_slow_ms=None,
            alpha=float(alpha),
            predicted=predicted.tolist(),
            residuals=residuals.tolist(),
        )

        if best is None or fit.bic < best.bic:
            best = fit

    if best is None:
        raise RuntimeError("No valid single-tau fit.")

    return best


def fit_two_tau_profile(
    *,
    observed: np.ndarray,
    tau_values: np.ndarray,
    tau_grid: np.ndarray,
    alpha_grid: np.ndarray,
    min_ratio: float,
    nonnegative_amplitudes: bool,
    ridge: float,
    alpha_count_as_parameter: bool,
) -> ProfileFit:
    best = None

    finite_count = int(np.isfinite(observed).sum())

    for fast_tau, slow_tau, alpha in itertools.product(
        tau_grid,
        tau_grid,
        alpha_grid,
    ):
        fast_tau = float(fast_tau)
        slow_tau = float(slow_tau)

        if slow_tau <= fast_tau:
            continue

        if slow_tau / fast_tau < min_ratio:
            continue

        fast_kernel = fcs_kernel(
            tau_values,
            tau_d_ms=fast_tau,
            alpha=float(alpha),
        )

        slow_kernel = fcs_kernel(
            tau_values,
            tau_d_ms=slow_tau,
            alpha=float(alpha),
        )

        coefficients, predicted, residuals, rss = solve_linear_profile(
            basis_columns=[fast_kernel, slow_kernel],
            observed=observed,
            nonnegative_amplitudes=nonnegative_amplitudes,
            ridge=ridge,
        )

        n_parameters = 5 + int(alpha_count_as_parameter)
        # baseline, two amplitudes, two tau_D values, optional alpha.

        rmse, aic, bic = calculate_ic(
            rss=rss,
            n_observations=finite_count,
            n_parameters=n_parameters,
        )

        fit = ProfileFit(
            model_type="two_tau",
            success=True,
            rss=float(rss),
            rmse=float(rmse),
            aic=float(aic),
            bic=float(bic),
            baseline=float(coefficients[0]),
            amplitude_fast=float(coefficients[1]),
            amplitude_slow=float(coefficients[2]),
            tau_d_fast_ms=fast_tau,
            tau_d_slow_ms=slow_tau,
            alpha=float(alpha),
            predicted=predicted.tolist(),
            residuals=residuals.tolist(),
        )

        if best is None or fit.bic < best.bic:
            best = fit

    if best is None:
        raise RuntimeError("No valid two-tau fit.")

    return best


def information_weight_percent(
    *,
    ic_a: float,
    ic_b: float,
    choose_b: bool,
) -> float:
    values = np.asarray([ic_a, ic_b], dtype=float)
    delta = values - np.nanmin(values)
    raw = np.exp(-0.5 * delta)
    weights = raw / raw.sum()

    return float(100.0 * (weights[1] if choose_b else weights[0]))


def decide_timepoint(
    *,
    row_index: int,
    time_value: float,
    single: ProfileFit,
    two: ProfileFit,
    min_delta_bic: float,
    min_delta_aic: float,
    min_rss_improvement_fraction: float,
    min_component_fraction: float,
    min_ratio: float,
) -> TimepointDecision:
    delta_bic = single.bic - two.bic
    delta_aic = single.aic - two.aic

    two_bic_weight = information_weight_percent(
        ic_a=single.bic,
        ic_b=two.bic,
        choose_b=True,
    )

    two_aic_weight = information_weight_percent(
        ic_a=single.aic,
        ic_b=two.aic,
        choose_b=True,
    )

    rss_improvement = (
        (single.rss - two.rss) / single.rss
        if single.rss > 0
        else 0.0
    )

    fast_amp = max(float(two.amplitude_fast or 0.0), 0.0)
    slow_amp = max(float(two.amplitude_slow or 0.0), 0.0)
    total_amp = fast_amp + slow_amp

    if total_amp > 0:
        fast_fraction = fast_amp / total_amp
        slow_fraction = slow_amp / total_amp
    else:
        fast_fraction = 0.0
        slow_fraction = 0.0

    ratio = (
        float(two.tau_d_slow_ms) / float(two.tau_d_fast_ms)
        if two.tau_d_fast_ms and two.tau_d_slow_ms
        else None
    )

    checks = []

    checks.append(
        (
            delta_bic >= min_delta_bic,
            f"delta_BIC {delta_bic:.3g} >= {min_delta_bic:.3g}",
        )
    )

    checks.append(
        (
            delta_aic >= min_delta_aic,
            f"delta_AIC {delta_aic:.3g} >= {min_delta_aic:.3g}",
        )
    )

    checks.append(
        (
            rss_improvement >= min_rss_improvement_fraction,
            f"RSS improvement {rss_improvement:.3g} >= {min_rss_improvement_fraction:.3g}",
        )
    )

    checks.append(
        (
            ratio is not None and ratio >= min_ratio,
            f"slow/fast ratio {ratio:.3g} >= {min_ratio:.3g}",
        )
    )

    checks.append(
        (
            fast_fraction >= min_component_fraction,
            f"fast amplitude fraction {fast_fraction:.3g} >= {min_component_fraction:.3g}",
        )
    )

    checks.append(
        (
            slow_fraction >= min_component_fraction,
            f"slow amplitude fraction {slow_fraction:.3g} >= {min_component_fraction:.3g}",
        )
    )

    passed = [ok for ok, _ in checks]

    two_supported = all(passed)

    if two_supported:
        selected = "two_tau"
        reason = "two_tau accepted: " + "; ".join(text for _, text in checks)
    else:
        selected = "single_tau"
        failed_text = "; ".join(text for ok, text in checks if not ok)
        reason = "single_tau selected because: " + failed_text

    return TimepointDecision(
        row_index=row_index,
        time_value=float(time_value),
        selected_model=selected,
        two_tau_supported=two_supported,
        reason=reason,
        single_rss=float(single.rss),
        single_bic=float(single.bic),
        single_aic=float(single.aic),
        single_tau_d_ms=float(single.tau_d_fast_ms),
        single_alpha=float(single.alpha),
        two_rss=float(two.rss),
        two_bic=float(two.bic),
        two_aic=float(two.aic),
        two_tau_d_fast_ms=float(two.tau_d_fast_ms),
        two_tau_d_slow_ms=float(two.tau_d_slow_ms),
        two_alpha=float(two.alpha),
        two_amplitude_fast=float(two.amplitude_fast),
        two_amplitude_slow=float(two.amplitude_slow),
        delta_bic_single_minus_two=float(delta_bic),
        delta_aic_single_minus_two=float(delta_aic),
        two_bic_weight_percent=float(two_bic_weight),
        two_aic_weight_percent=float(two_aic_weight),
        rss_improvement_fraction=float(rss_improvement),
        slow_fast_ratio=float(ratio) if ratio is not None else None,
        fast_amplitude_fraction=float(fast_fraction),
        slow_amplitude_fraction=float(slow_fraction),
    )


def summarize_decisions(decisions: list[TimepointDecision]) -> dict:
    n = len(decisions)
    n_two = sum(decision.two_tau_supported for decision in decisions)
    fraction_two = n_two / n if n else 0.0

    two_times = [
        decision.time_value
        for decision in decisions
        if decision.two_tau_supported
    ]

    median_delta_bic = float(
        np.nanmedian(
            [
                decision.delta_bic_single_minus_two
                for decision in decisions
                if decision.delta_bic_single_minus_two is not None
            ]
        )
    )

    median_two_weight = float(
        np.nanmedian(
            [
                decision.two_bic_weight_percent
                for decision in decisions
                if decision.two_bic_weight_percent is not None
            ]
        )
    )

    if fraction_two >= 0.6:
        recommendation = "two_tau_surface"
    elif fraction_two <= 0.25:
        recommendation = "single_tau_surface"
    else:
        recommendation = "mixed_or_profiled_tau_surface"

    return {
        "n_timepoints": n,
        "n_two_tau_supported": int(n_two),
        "fraction_two_tau_supported": float(fraction_two),
        "two_tau_supported_times": two_times,
        "median_delta_bic_single_minus_two": median_delta_bic,
        "median_two_tau_bic_weight_percent": median_two_weight,
        "recommendation": recommendation,
    }


def plot_decision_summary(
    *,
    decisions: list[TimepointDecision],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []

    times = np.asarray([decision.time_value for decision in decisions], dtype=float)
    delta_bic = np.asarray(
        [decision.delta_bic_single_minus_two for decision in decisions],
        dtype=float,
    )
    two_weights = np.asarray(
        [decision.two_bic_weight_percent for decision in decisions],
        dtype=float,
    )
    selected = np.asarray(
        [1.0 if decision.two_tau_supported else 0.0 for decision in decisions],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0.0, linewidth=1.0)
    ax.plot(times, delta_bic, marker="o")
    ax.set_title("Two-tau support over elapsed time")
    ax.set_xlabel("time")
    ax.set_ylabel("Delta BIC = BIC(single) - BIC(two)")
    fig.tight_layout()
    path = output_dir / "delta_bic_two_tau_support_by_time.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(times, two_weights, marker="o")
    ax.set_title("Two-tau BIC weight over elapsed time")
    ax.set_xlabel("time")
    ax.set_ylabel("two-tau BIC weight (%)")
    ax.set_ylim(-2, 102)
    fig.tight_layout()
    path = output_dir / "two_tau_bic_weight_by_time.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.step(times, selected, where="mid")
    ax.set_title("Selected FCS profile equation by timepoint")
    ax.set_xlabel("time")
    ax.set_ylabel("selected")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["single_tau", "two_tau"])
    fig.tight_layout()
    path = output_dir / "selected_tau_model_by_time.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    written.append(path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select single- vs two-tau_D FCS profile equations per timepoint."
        )
    )

    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--time-column", default="time_min")

    parser.add_argument("--tau-min-ms", type=float, default=1e-3)
    parser.add_argument("--tau-max-ms", type=float, default=1e5)
    parser.add_argument("--n-tau-grid", type=int, default=32)
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=[1.0])

    parser.add_argument("--min-ratio", type=float, default=10.0)
    parser.add_argument("--min-delta-bic", type=float, default=10.0)
    parser.add_argument("--min-delta-aic", type=float, default=4.0)
    parser.add_argument("--min-rss-improvement-fraction", type=float, default=0.02)
    parser.add_argument("--min-component-fraction", type=float, default=0.05)

    parser.add_argument("--ridge", type=float, default=1e-10)
    parser.add_argument("--allow-negative-amplitudes", action="store_true")
    parser.add_argument("--max-timepoints", type=int, default=None)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe, tau_values, signal_columns, matrix = load_fcs_profiles(
        data_path=Path(args.data),
        time_column=args.time_column,
    )

    time_values = dataframe[args.time_column].to_numpy(dtype=float)

    if args.max_timepoints is not None:
        matrix = matrix[: args.max_timepoints]
        time_values = time_values[: args.max_timepoints]

    tau_grid = np.logspace(
        np.log10(args.tau_min_ms),
        np.log10(args.tau_max_ms),
        args.n_tau_grid,
    )

    alpha_grid = np.asarray(args.alpha_grid, dtype=float)
    alpha_count_as_parameter = len(alpha_grid) > 1

    print("\nFCS tau-model selection by timepoint")
    print("====================================")
    print(f"Data: {args.data}")
    print(f"Timepoints: {len(time_values)}")
    print(f"Tau points: {len(tau_values)}")
    print(f"Tau grid points: {len(tau_grid)}")
    print(f"Alpha grid: {alpha_grid}")
    print(f"Min tau ratio: {args.min_ratio}")

    decisions: list[TimepointDecision] = []
    single_fits = []
    two_fits = []

    for index, observed in enumerate(matrix):
        time_value = float(time_values[index])

        print(
            f"Fitting timepoint {index + 1}/{len(time_values)} "
            f"time={time_value:g}",
            flush=True,
        )

        single = fit_single_tau_profile(
            observed=observed,
            tau_values=tau_values,
            tau_grid=tau_grid,
            alpha_grid=alpha_grid,
            nonnegative_amplitudes=not args.allow_negative_amplitudes,
            ridge=args.ridge,
            alpha_count_as_parameter=alpha_count_as_parameter,
        )

        two = fit_two_tau_profile(
            observed=observed,
            tau_values=tau_values,
            tau_grid=tau_grid,
            alpha_grid=alpha_grid,
            min_ratio=args.min_ratio,
            nonnegative_amplitudes=not args.allow_negative_amplitudes,
            ridge=args.ridge,
            alpha_count_as_parameter=alpha_count_as_parameter,
        )

        decision = decide_timepoint(
            row_index=index,
            time_value=time_value,
            single=single,
            two=two,
            min_delta_bic=args.min_delta_bic,
            min_delta_aic=args.min_delta_aic,
            min_rss_improvement_fraction=args.min_rss_improvement_fraction,
            min_component_fraction=args.min_component_fraction,
            min_ratio=args.min_ratio,
        )

        decisions.append(decision)
        single_fits.append(single)
        two_fits.append(two)

        print(
            f"  selected={decision.selected_model} "
            f"delta_BIC={decision.delta_bic_single_minus_two:.3g} "
            f"two_weight={decision.two_bic_weight_percent:.2f}% "
            f"reason={decision.reason}",
            flush=True,
        )

    decisions_dataframe = pd.DataFrame([asdict(decision) for decision in decisions])
    decisions_path = output_dir / "fcs_tau_model_decisions_by_timepoint.csv"
    decisions_dataframe.to_csv(decisions_path, index=False)

    summary = summarize_decisions(decisions)

    summary_path = output_dir / "fcs_tau_model_selection_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    fits_payload = {
        "single_fits": [asdict(fit) for fit in single_fits],
        "two_fits": [asdict(fit) for fit in two_fits],
    }

    fits_path = output_dir / "fcs_tau_model_profile_fits.json"
    with fits_path.open("w") as handle:
        json.dump(fits_payload, handle)

    figure_paths = plot_decision_summary(
        decisions=decisions,
        output_dir=output_dir / "figures",
    )

    print("\nTau-model selection complete")
    print("============================")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote decisions: {decisions_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote fits: {fits_path}")
    print("Figures:")
    for path in figure_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()

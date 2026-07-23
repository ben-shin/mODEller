from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def load_run_config(profile_dir: Path) -> dict:
    path = profile_dir / "species_tau_pair_profile_config.json"

    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")

    return load_json(path)


def load_best_table(profile_dir: Path) -> pd.DataFrame:
    path = profile_dir / "species_tau_pair_profile_best_by_model.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing best table: {path}")

    return pd.read_csv(path)


def get_surface_matrix(
    dataframe: pd.DataFrame,
    *,
    time_column: str,
):
    if time_column not in dataframe.columns:
        if "time_min" in dataframe.columns:
            time_column = "time_min"
        elif "time" in dataframe.columns:
            time_column = "time"
        else:
            raise ValueError(
                f"Could not find time column {time_column!r}. "
                f"Columns include: {list(dataframe.columns[:10])}"
            )

    signal_columns = [
        column
        for column in dataframe.columns
        if column != time_column
    ]

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
    matrix = dataframe[signal_columns].to_numpy(dtype=float)

    return time_values, tau_values, signal_columns, matrix


def choose_evenly_spaced_indices(n_items: int, n_select: int) -> list[int]:
    if n_items <= 0:
        return []

    if n_items <= n_select:
        return list(range(n_items))

    values = np.linspace(0, n_items - 1, n_select)

    return sorted(set(int(round(value)) for value in values))


def plot_surface_heatmap(
    *,
    matrix: np.ndarray,
    time_values: np.ndarray,
    tau_values: np.ndarray,
    output_path: Path,
    title: str,
    colorbar_label: str,
):
    fig, ax = plt.subplots(figsize=(9, 6))

    if np.all(tau_values > 0):
        extent = [
            np.log10(float(np.nanmin(tau_values))),
            np.log10(float(np.nanmax(tau_values))),
            float(np.nanmin(time_values)),
            float(np.nanmax(time_values)),
        ]
        xlabel = "log10(tau / ms)"
    else:
        extent = [
            0,
            matrix.shape[1] - 1,
            float(np.nanmin(time_values)),
            float(np.nanmax(time_values)),
        ]
        xlabel = "tau index"

    image = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=extent,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("time_min")

    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_profiles_observed_vs_fitted(
    *,
    observed_matrix: np.ndarray,
    predicted_matrix: np.ndarray,
    time_values: np.ndarray,
    tau_values: np.ndarray,
    model_name: str,
    output_path: Path,
    n_profiles: int,
):
    indices = choose_evenly_spaced_indices(
        len(time_values),
        n_profiles,
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    for index in indices:
        ax.plot(
            tau_values,
            observed_matrix[index, :],
            marker="o",
            markersize=2,
            linewidth=0.8,
            alpha=0.55,
            label=f"{time_values[index]:g} min observed",
        )

        ax.plot(
            tau_values,
            predicted_matrix[index, :],
            linewidth=1.4,
            alpha=0.95,
            label=f"{time_values[index]:g} min fitted",
        )

    if np.all(tau_values > 0):
        ax.set_xscale("log")

    ax.set_title(f"{model_name}: observed vs fitted FCS profiles")
    ax.set_xlabel("tau / ms [log]")
    ax.set_ylabel("G(tau)")

    if len(indices) <= 5:
        ax.legend(fontsize="x-small")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_each_timepoint_profile(
    *,
    observed_matrix: np.ndarray,
    predicted_matrix: np.ndarray,
    time_values: np.ndarray,
    tau_values: np.ndarray,
    model_name: str,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []

    for index, time_value in enumerate(time_values):
        fig, ax = plt.subplots(figsize=(7, 4.5))

        ax.plot(
            tau_values,
            observed_matrix[index, :],
            marker="o",
            markersize=2,
            linewidth=0.9,
            label="observed",
        )

        ax.plot(
            tau_values,
            predicted_matrix[index, :],
            linewidth=1.8,
            label="fitted",
        )

        if np.all(tau_values > 0):
            ax.set_xscale("log")

        ax.set_title(f"{model_name}: time = {time_value:g} min")
        ax.set_xlabel("tau / ms [log]")
        ax.set_ylabel("G(tau)")
        ax.legend()

        fig.tight_layout()

        path = output_dir / f"time_{time_value:g}_min_observed_vs_fitted.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)

        written.append(path)

    return written


def plot_timecourses_at_selected_tau(
    *,
    observed_matrix: np.ndarray,
    predicted_matrix: np.ndarray,
    time_values: np.ndarray,
    tau_values: np.ndarray,
    model_name: str,
    output_path: Path,
    n_tau: int,
):
    indices = choose_evenly_spaced_indices(
        len(tau_values),
        n_tau,
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    for index in indices:
        ax.plot(
            time_values,
            observed_matrix[:, index],
            marker="o",
            markersize=2,
            linewidth=0.8,
            alpha=0.6,
            label=f"tau={tau_values[index]:.3g} ms observed",
        )

        ax.plot(
            time_values,
            predicted_matrix[:, index],
            linewidth=1.4,
            alpha=0.95,
            label=f"tau={tau_values[index]:.3g} ms fitted",
        )

    ax.set_title(f"{model_name}: observed vs fitted timecourses at selected tau")
    ax.set_xlabel("time_min")
    ax.set_ylabel("G(tau, time)")

    if len(indices) <= 5:
        ax.legend(fontsize="x-small")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_component_timecourses(
    *,
    model_dir: Path,
    model_name: str,
    output_path: Path,
):
    path = model_dir / "species_component_timecourses.csv"

    if not path.exists():
        return None

    dataframe = pd.read_csv(path)

    if "time_min" not in dataframe.columns:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    for column in [
        "fast_species_timecourse",
        "slow_species_timecourse",
    ]:
        if column in dataframe.columns:
            ax.plot(
                dataframe["time_min"].to_numpy(dtype=float),
                dataframe[column].to_numpy(dtype=float),
                marker="o",
                markersize=3,
                linewidth=1.3,
                label=column,
            )

    ax.set_title(f"{model_name}: ODE-derived fast/slow component timecourses")
    ax.set_xlabel("time_min")
    ax.set_ylabel("component abundance")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def plot_one_model(
    *,
    profile_dir: Path,
    model_name: str,
    observed_time: np.ndarray,
    observed_tau: np.ndarray,
    observed_matrix: np.ndarray,
    figures_dir: Path,
    n_profiles: int,
    n_tau_timecourses: int,
    plot_all_timepoints: bool,
):
    model_dir = profile_dir / "models" / safe_name(model_name)

    predicted_path = model_dir / "species_surface_predicted.csv"
    residuals_path = model_dir / "species_surface_residuals.csv"

    if not predicted_path.exists():
        raise FileNotFoundError(
            f"Missing predicted surface for {model_name}: {predicted_path}. "
            "Did you run the fitter with --save-best-surfaces?"
        )

    predicted_dataframe = pd.read_csv(predicted_path)
    residuals_dataframe = pd.read_csv(residuals_path)

    predicted_time, predicted_tau, _, predicted_matrix = get_surface_matrix(
        predicted_dataframe,
        time_column="time_min",
    )

    residual_time, residual_tau, _, residual_matrix = get_surface_matrix(
        residuals_dataframe,
        time_column="time_min",
    )

    if predicted_matrix.shape != observed_matrix.shape:
        raise ValueError(
            f"Shape mismatch for {model_name}: "
            f"observed={observed_matrix.shape}, predicted={predicted_matrix.shape}"
        )

    model_figures_dir = figures_dir / safe_name(model_name)
    model_figures_dir.mkdir(parents=True, exist_ok=True)

    written = []

    path = model_figures_dir / "predicted_surface_heatmap.png"
    plot_surface_heatmap(
        matrix=predicted_matrix,
        time_values=predicted_time,
        tau_values=predicted_tau,
        output_path=path,
        title=f"{model_name}: predicted FCS surface",
        colorbar_label="predicted G",
    )
    written.append(path)

    path = model_figures_dir / "residual_surface_heatmap.png"
    plot_surface_heatmap(
        matrix=residual_matrix,
        time_values=residual_time,
        tau_values=residual_tau,
        output_path=path,
        title=f"{model_name}: residual surface",
        colorbar_label="observed - fitted",
    )
    written.append(path)

    path = model_figures_dir / "profiles_observed_vs_fitted.png"
    plot_profiles_observed_vs_fitted(
        observed_matrix=observed_matrix,
        predicted_matrix=predicted_matrix,
        time_values=observed_time,
        tau_values=observed_tau,
        model_name=model_name,
        output_path=path,
        n_profiles=n_profiles,
    )
    written.append(path)

    path = model_figures_dir / "timecourses_at_selected_tau_observed_vs_fitted.png"
    plot_timecourses_at_selected_tau(
        observed_matrix=observed_matrix,
        predicted_matrix=predicted_matrix,
        time_values=observed_time,
        tau_values=observed_tau,
        model_name=model_name,
        output_path=path,
        n_tau=n_tau_timecourses,
    )
    written.append(path)

    component_path = plot_component_timecourses(
        model_dir=model_dir,
        model_name=model_name,
        output_path=model_figures_dir / "species_component_timecourses.png",
    )

    if component_path is not None:
        written.append(component_path)

    if plot_all_timepoints:
        written.extend(
            plot_each_timepoint_profile(
                observed_matrix=observed_matrix,
                predicted_matrix=predicted_matrix,
                time_values=observed_time,
                tau_values=observed_tau,
                model_name=model_name,
                output_dir=model_figures_dir / "all_timepoint_profiles",
            )
        )

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot observed vs fitted curves for species-group FCS surface fits."
        )
    )

    parser.add_argument(
        "--profile-dir",
        required=True,
        help="Output directory from profile_fcs_surface_species_tau_pairs.py.",
    )

    parser.add_argument(
        "--figures-dir",
        default=None,
        help="Defaults to <profile-dir>/fit_curve_figures.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Number of best models to plot. Default: all models with saved surfaces.",
    )

    parser.add_argument(
        "--sort-by",
        default="bic",
    )

    parser.add_argument(
        "--n-profiles",
        type=int,
        default=6,
        help="Number of elapsed timepoints to overlay in summary profile plot.",
    )

    parser.add_argument(
        "--n-tau-timecourses",
        type=int,
        default=6,
        help="Number of tau values to overlay in timecourse plot.",
    )

    parser.add_argument(
        "--all-timepoints",
        action="store_true",
        help="Also save one observed-vs-fitted G(tau) plot per elapsed timepoint.",
    )

    args = parser.parse_args()

    profile_dir = Path(args.profile_dir)
    figures_dir = (
        Path(args.figures_dir)
        if args.figures_dir is not None
        else profile_dir / "fit_curve_figures"
    )
    figures_dir.mkdir(parents=True, exist_ok=True)

    config = load_run_config(profile_dir)
    data_path = Path(config["data"])
    time_column = config.get("time_column", "time_min")

    observed_dataframe = pd.read_csv(data_path)

    observed_time, observed_tau, _, observed_matrix = get_surface_matrix(
        observed_dataframe,
        time_column=time_column,
    )

    best_table = load_best_table(profile_dir)

    if args.sort_by in best_table.columns:
        best_table = best_table.sort_values(args.sort_by, na_position="last")

    if args.top_n is not None:
        best_table = best_table.head(args.top_n)

    model_names = [str(name) for name in best_table["model_name"]]

    written = []

    observed_path = figures_dir / "observed_surface_heatmap.png"
    plot_surface_heatmap(
        matrix=observed_matrix,
        time_values=observed_time,
        tau_values=observed_tau,
        output_path=observed_path,
        title="Observed raw FCS surface",
        colorbar_label="observed G",
    )
    written.append(observed_path)

    for model_name in model_names:
        try:
            written.extend(
                plot_one_model(
                    profile_dir=profile_dir,
                    model_name=model_name,
                    observed_time=observed_time,
                    observed_tau=observed_tau,
                    observed_matrix=observed_matrix,
                    figures_dir=figures_dir,
                    n_profiles=args.n_profiles,
                    n_tau_timecourses=args.n_tau_timecourses,
                    plot_all_timepoints=args.all_timepoints,
                )
            )

        except FileNotFoundError as exc:
            print(f"Skipping {model_name}: {exc}", flush=True)

    index_path = figures_dir / "fit_curve_figure_index.txt"

    with index_path.open("w") as handle:
        handle.write("Species-group FCS surface fit curve figures\n")
        handle.write("===========================================\n\n")
        handle.write(f"Profile dir: {profile_dir}\n")
        handle.write(f"Data: {data_path}\n")
        handle.write(f"Models requested: {', '.join(model_names)}\n\n")

        for path in sorted(written):
            try:
                relative = path.relative_to(figures_dir)
            except ValueError:
                relative = path

            handle.write(f"- {relative}\n")

    written.append(index_path)

    print("\nSpecies-group FCS fitted-curve plots complete")
    print("=============================================")
    print(f"Profile dir: {profile_dir}")
    print(f"Figures dir: {figures_dir}")
    print(f"Models plotted: {', '.join(model_names)}")
    print(f"Files written: {len(written)}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()

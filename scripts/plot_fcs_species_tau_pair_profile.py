from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_heatmap(
    *,
    dataframe: pd.DataFrame,
    model_name: str,
    value_column: str,
    output_path: Path,
):
    subset = dataframe[dataframe["model_name"].astype(str) == model_name].copy()
    subset[value_column] = pd.to_numeric(subset[value_column], errors="coerce")
    subset = subset.dropna(subset=[value_column])

    if subset.empty:
        return

    fast_values = sorted(subset["tau_d_fast_ms"].astype(float).unique())
    slow_values = sorted(subset["tau_d_slow_ms"].astype(float).unique())

    matrix = np.full((len(slow_values), len(fast_values)), np.nan)

    fast_index = {value: index for index, value in enumerate(fast_values)}
    slow_index = {value: index for index, value in enumerate(slow_values)}

    for _, row in subset.iterrows():
        fast = float(row["tau_d_fast_ms"])
        slow = float(row["tau_d_slow_ms"])
        matrix[slow_index[slow], fast_index[fast]] = float(row[value_column])

    fig, ax = plt.subplots(figsize=(8, 6))

    image = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_title(f"{model_name}: species-group tau pair profile by {value_column}")
    ax.set_xlabel("fast tau_D / ms")
    ax.set_ylabel("slow tau_D / ms")
    ax.set_xticks(np.arange(len(fast_values)))
    ax.set_xticklabels([f"{value:.2g}" for value in fast_values], rotation=45)
    ax.set_yticks(np.arange(len(slow_values)))
    ax.set_yticklabels([f"{value:.2g}" for value in slow_values])

    fig.colorbar(image, ax=ax, label=value_column)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_best_by_model(
    *,
    best: pd.DataFrame,
    output_dir: Path,
):
    written = []

    if best.empty:
        return written

    for value_column in ["bic", "aic", "rss", "bic_weight_percent"]:
        if value_column not in best.columns:
            continue

        subset = best.copy()
        subset[value_column] = pd.to_numeric(subset[value_column], errors="coerce")
        subset = subset.dropna(subset=[value_column])

        if subset.empty:
            continue

        if value_column == "bic_weight_percent":
            subset = subset.sort_values(value_column, ascending=False)
            ylabel = "BIC weight (%)"
        else:
            subset = subset.sort_values(value_column)
            ylabel = value_column.upper()

        fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(subset)), 5))

        ax.bar(
            subset["model_name"].astype(str),
            subset[value_column].astype(float),
        )

        ax.set_title(f"Best species-group tau-pair result by model: {value_column}")
        ax.set_xlabel("model")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)

        fig.tight_layout()

        path = output_dir / f"best_by_model_{value_column}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path)

    return written


def plot_component_fractions(
    *,
    best: pd.DataFrame,
    output_dir: Path,
):
    required = {
        "model_name",
        "fast_amplitude_fraction",
        "slow_amplitude_fraction",
    }

    if best.empty or not required.issubset(best.columns):
        return []

    subset = best.copy()

    x = np.arange(len(subset))

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(subset)), 5))

    ax.bar(
        x,
        subset["fast_amplitude_fraction"].astype(float),
        label="fast component",
    )

    ax.bar(
        x,
        subset["slow_amplitude_fraction"].astype(float),
        bottom=subset["fast_amplitude_fraction"].astype(float),
        label="slow component",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(subset["model_name"].astype(str), rotation=45)
    ax.set_ylim(0, 1)
    ax.set_title("Best-fit component amplitude fractions")
    ax.set_xlabel("model")
    ax.set_ylabel("amplitude fraction")
    ax.legend()

    fig.tight_layout()

    path = output_dir / "component_amplitude_fractions_by_model.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    return [path]


def plot_tau_pairs(
    *,
    best: pd.DataFrame,
    output_dir: Path,
):
    required = {"model_name", "tau_d_fast_ms", "tau_d_slow_ms"}

    if best.empty or not required.issubset(best.columns):
        return []

    subset = best.copy()
    x = np.arange(len(subset))

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(subset)), 5))

    ax.scatter(
        x,
        subset["tau_d_fast_ms"].astype(float),
        label="fast tau_D",
    )

    ax.scatter(
        x,
        subset["tau_d_slow_ms"].astype(float),
        label="slow tau_D",
    )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(subset["model_name"].astype(str), rotation=45)
    ax.set_title("Best species-group tau_D pair by model")
    ax.set_xlabel("model")
    ax.set_ylabel("tau_D / ms [log]")
    ax.legend()

    fig.tight_layout()

    path = output_dir / "best_tau_d_pair_by_model.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    return [path]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot species-group two-tau_D FCS profile outputs."
    )

    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--value-column", default="bic")

    args = parser.parse_args()

    profile_dir = Path(args.profile_dir)
    output_dir = Path(args.output_dir) if args.output_dir else profile_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_path = profile_dir / "species_tau_pair_profile.csv"
    best_path = profile_dir / "species_tau_pair_profile_best_by_model.csv"

    dataframe = pd.read_csv(profile_path)
    best = pd.read_csv(best_path) if best_path.exists() else pd.DataFrame()

    written = []

    successful = dataframe[dataframe["success"].astype(bool)].copy()

    for model_name in sorted(successful["model_name"].astype(str).unique()):
        path = output_dir / f"{model_name}_species_tau_pair_{args.value_column}_heatmap.png"

        plot_heatmap(
            dataframe=successful,
            model_name=model_name,
            value_column=args.value_column,
            output_path=path,
        )

        if path.exists():
            written.append(path)

    written.extend(plot_best_by_model(best=best, output_dir=output_dir))
    written.extend(plot_tau_pairs(best=best, output_dir=output_dir))
    written.extend(plot_component_fractions(best=best, output_dir=output_dir))

    index_path = output_dir / "species_tau_pair_figure_index.txt"

    with index_path.open("w") as handle:
        handle.write("Species-group tau-pair profile figures\n")
        handle.write("======================================\n\n")

        for path in sorted(written):
            handle.write(f"- {path.name}\n")

    print("\nSpecies-group tau-pair figures complete")
    print("=======================================")
    print(f"Profile dir: {profile_dir}")
    print(f"Figures dir: {output_dir}")
    print(f"Figures written: {len(written)}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def load_config(profile_dir: Path) -> dict:
    for name in ["species_tau_pair_profile_config.json", "parallel_profile_config.json"]:
        path = profile_dir / name
        if path.exists():
            return load_json(path)

    raise FileNotFoundError("No profile config found.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", required=True)
    args = parser.parse_args()

    profile_dir = Path(args.profile_dir)
    config = load_config(profile_dir)

    data_path = Path(config["data"])
    time_column = config.get("time_column", "time_min")

    observed = pd.read_csv(data_path)
    best = pd.read_csv(profile_dir / "species_tau_pair_profile_best_by_model.csv")

    signal_columns = [
        column for column in observed.columns
        if column != time_column and str(column).startswith("G_tau_")
    ]

    observed_matrix = observed[signal_columns].to_numpy(dtype=float)

    rows = []

    for model_name in best["model_name"].astype(str):
        pred_path = (
            profile_dir
            / "models"
            / safe_name(model_name)
            / "species_surface_predicted.csv"
        )

        if not pred_path.exists():
            rows.append(
                {
                    "model_name": model_name,
                    "status": "missing_predicted_csv",
                }
            )
            continue

        predicted = pd.read_csv(pred_path)
        pred_columns = [column for column in predicted.columns if column != "time_min"]
        predicted_matrix = predicted[pred_columns].to_numpy(dtype=float)

        residual = observed_matrix - predicted_matrix

        rows.append(
            {
                "model_name": model_name,
                "status": "ok",
                "observed_min": float(np.nanmin(observed_matrix)),
                "observed_max": float(np.nanmax(observed_matrix)),
                "predicted_min": float(np.nanmin(predicted_matrix)),
                "predicted_max": float(np.nanmax(predicted_matrix)),
                "max_abs_residual": float(np.nanmax(np.abs(residual))),
                "rmse": float(np.sqrt(np.nanmean(residual ** 2))),
            }
        )

    out = pd.DataFrame(rows)
    out_path = profile_dir / "fit_quality_summary.csv"
    out.to_csv(out_path, index=False)

    print(out.to_string(index=False))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()

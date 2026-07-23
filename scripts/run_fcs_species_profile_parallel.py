from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


def compute_weights(dataframe: pd.DataFrame, criterion: str) -> pd.DataFrame:
    output = dataframe.copy()

    if criterion not in output.columns or output.empty:
        return output

    values = pd.to_numeric(output[criterion], errors="coerce")

    if "success" in output.columns:
        success = output["success"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        success = pd.Series(True, index=output.index)

    valid = values.notna() & success

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


def model_paths(model_dir: Path) -> list[Path]:
    paths = sorted(model_dir.glob("*.txt"))

    return [
        path
        for path in paths
        if path.is_file()
    ]


def make_single_model_dir(
    *,
    source_model_path: Path,
    root: Path,
) -> Path:
    single_dir = root / source_model_path.stem
    single_dir.mkdir(parents=True, exist_ok=True)

    destination = single_dir / source_model_path.name
    shutil.copy2(source_model_path, destination)

    return single_dir


def build_command(
    *,
    python_executable: str,
    script_path: Path,
    args,
    single_model_dir: Path,
    single_output_dir: Path,
) -> list[str]:
    command = [
        python_executable,
        str(script_path),
        "--data",
        args.data,
        "--model-dir",
        str(single_model_dir),
        "--output-dir",
        str(single_output_dir),
        "--engine-name",
        args.engine_name,
        "--time-column",
        args.time_column,
        "--tau-min-ms",
        str(args.tau_min_ms),
        "--tau-max-ms",
        str(args.tau_max_ms),
        "--n-tau-grid",
        str(args.n_tau_grid),
        "--min-ratio",
        str(args.min_ratio),
        "--alpha",
        str(args.alpha),
        "--triplet-fraction",
        str(args.triplet_fraction),
        "--triplet-tau-ms",
        str(args.triplet_tau_ms),
        "--initial-guess",
        str(args.initial_guess),
        "--lower-bound",
        str(args.lower_bound),
        "--upper-bound",
        str(args.upper_bound),
        "--max-nfev",
        str(args.max_nfev),
        "--rtol",
        str(args.rtol),
        "--atol",
        str(args.atol),
        "--ridge",
        str(args.ridge),
    ]

    if args.fast_max_index is not None:
        command.extend(["--fast-max-index", str(args.fast_max_index)])

    if args.initial_active_species:
        command.extend(["--initial-active-species", args.initial_active_species])

    if args.fast_species:
        command.append("--fast-species")
        command.extend(args.fast_species)

    if args.slow_species:
        command.append("--slow-species")
        command.extend(args.slow_species)
    
    if args.max_predicted_g is not None:
        command.extend(["--max-predicted-g", str(args.max_predicted_g)])

        command.extend(
            [
                "--prediction-ceiling-penalty",
                str(args.prediction_ceiling_penalty),
            ]
        )

    if args.allow_negative_components:
        command.append("--allow-negative-components")

    if args.save_best_surfaces:
        command.append("--save-best-surfaces")

    return command


def run_one_model(
    *,
    model_path: Path,
    args,
    script_path: Path,
    single_model_root: Path,
    per_model_output_root: Path,
    logs_dir: Path,
) -> dict:
    model_name = model_path.stem

    single_model_dir = make_single_model_dir(
        source_model_path=model_path,
        root=single_model_root,
    )

    single_output_dir = per_model_output_root / model_name
    single_output_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / f"{model_name}.log"

    command = build_command(
        python_executable=args.python_executable,
        script_path=script_path,
        args=args,
        single_model_dir=single_model_dir,
        single_output_dir=single_output_dir,
    )

    env = os.environ.copy()

    # Avoid each subprocess spawning many BLAS/OpenMP threads.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")

    with log_path.open("w") as handle:
        handle.write("COMMAND\n")
        handle.write(" ".join(command) + "\n\n")
        handle.flush()

        result = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )

    return {
        "model_name": model_name,
        "returncode": result.returncode,
        "output_dir": str(single_output_dir),
        "log_path": str(log_path),
        "command": command,
    }


def merge_outputs(
    *,
    output_dir: Path,
    per_model_output_root: Path,
    results: list[dict],
) -> None:
    profile_frames = []
    best_frames = []
    preview_frames = []

    models_output_dir = output_dir / "models"
    models_output_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        model_name = result["model_name"]
        single_output_dir = Path(result["output_dir"])

        profile_path = single_output_dir / "species_tau_pair_profile.csv"
        best_path = single_output_dir / "species_tau_pair_profile_best_by_model.csv"
        preview_path = single_output_dir / "species_group_preview.csv"

        if profile_path.exists():
            profile_frames.append(pd.read_csv(profile_path))

        if best_path.exists():
            best_frames.append(pd.read_csv(best_path))

        if preview_path.exists():
            preview_frames.append(pd.read_csv(preview_path))

        single_models_dir = single_output_dir / "models"

        if single_models_dir.exists():
            for model_surface_dir in single_models_dir.iterdir():
                if not model_surface_dir.is_dir():
                    continue

                destination = models_output_dir / model_surface_dir.name

                if destination.exists():
                    shutil.rmtree(destination)

                shutil.copytree(model_surface_dir, destination)

    if profile_frames:
        profile = pd.concat(profile_frames, ignore_index=True)
        profile = compute_weights(profile, "bic")
        profile = compute_weights(profile, "aic")

        if "bic" in profile.columns:
            profile = profile.sort_values("bic", na_position="last")

        profile.to_csv(output_dir / "species_tau_pair_profile.csv", index=False)

    if best_frames:
        best = pd.concat(best_frames, ignore_index=True)
        best = compute_weights(best, "bic")
        best = compute_weights(best, "aic")

        if "bic" in best.columns:
            best = best.sort_values("bic", na_position="last")

        best.to_csv(output_dir / "species_tau_pair_profile_best_by_model.csv", index=False)

    if preview_frames:
        preview = pd.concat(preview_frames, ignore_index=True)
        preview.to_csv(output_dir / "species_group_preview.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run species-group FCS tau-pair profiling in parallel across models."
    )

    parser.add_argument("--data", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--engine-name", default="numba_projection")
    parser.add_argument("--time-column", default="time_min")

    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--python-executable", default=sys.executable)

    parser.add_argument("--fast-species", nargs="+", default=None)
    parser.add_argument("--slow-species", nargs="+", default=None)
    parser.add_argument("--fast-max-index", type=int, default=3)
    parser.add_argument("--initial-active-species", default=None)

    parser.add_argument("--tau-min-ms", type=float, default=1e-5)
    parser.add_argument("--tau-max-ms", type=float, default=1e5)
    parser.add_argument("--n-tau-grid", type=int, default=8)
    parser.add_argument("--min-ratio", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=1.0)

    parser.add_argument("--triplet-fraction", type=float, default=0.0)
    parser.add_argument("--triplet-tau-ms", type=float, default=1.0)
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
    parser.add_argument("--save-best-surfaces", action="store_true")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    single_model_root = output_dir / "_single_model_inputs"
    single_model_root.mkdir(parents=True, exist_ok=True)

    per_model_output_root = output_dir / "_per_model_outputs"
    per_model_output_root.mkdir(parents=True, exist_ok=True)

    script_path = Path("scripts/profile_fcs_surface_species_tau_pairs.py")

    models = model_paths(Path(args.model_dir))

    if not models:
        raise ValueError(f"No .txt model files found in {args.model_dir}")

    with (output_dir / "parallel_profile_config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2)

    print("\nParallel species-group FCS profile")
    print("==================================")
    print(f"Models: {len(models)}")
    print(f"Workers: {args.n_workers}")
    print(f"Output dir: {output_dir}")

    results = []

    with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {
            executor.submit(
                run_one_model,
                model_path=model_path,
                args=args,
                script_path=script_path,
                single_model_root=single_model_root,
                per_model_output_root=per_model_output_root,
                logs_dir=logs_dir,
            ): model_path
            for model_path in models
        }

        for future in as_completed(futures):
            model_path = futures[future]
            result = future.result()
            results.append(result)

            status = "PASS" if result["returncode"] == 0 else "FAIL"

            print(
                f"{status}: {model_path.stem} "
                f"log={result['log_path']}",
                flush=True,
            )

    results = sorted(results, key=lambda item: item["model_name"])

    with (output_dir / "parallel_profile_results.json").open("w") as handle:
        json.dump(results, handle, indent=2)

    merge_outputs(
        output_dir=output_dir,
        per_model_output_root=per_model_output_root,
        results=results,
    )

    failed = [
        result
        for result in results
        if result["returncode"] != 0
    ]

    print("\nParallel profile complete")
    print("=========================")
    print(f"Output dir: {output_dir}")
    print(f"Failed models: {len(failed)}")

    if failed:
        print("\nFailures:")
        for result in failed:
            print(f"  {result['model_name']}: {result['log_path']}")

    best_path = output_dir / "species_tau_pair_profile_best_by_model.csv"

    if best_path.exists():
        best = pd.read_csv(best_path)
        cols = [
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
        ]
        cols = [col for col in cols if col in best.columns]

        print("\nBest by model:")
        print(best[cols].to_string(index=False))


if __name__ == "__main__":
    main()

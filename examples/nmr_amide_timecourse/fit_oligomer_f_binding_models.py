from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from odefit.data.dataset import Dataset
from odefit.export.bundle_export import export_fit_bundle
from odefit.fitting.fit_result import FitResult
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.model_comparison import build_ranked_model_comparison_table
from odefit.fitting.observable_spec import ObservableSpec
from odefit.fitting.optimizer import fit_model
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import ModelSpec, build_model_spec


EXAMPLE_DIR = Path(__file__).parent
REPO_ROOT = EXAMPLE_DIR.parents[1]
OUTPUT_DIR = EXAMPLE_DIR / "oligomer_f_binding_outputs"
BALANCED_MONOMER_ONLY_OUTPUT_DIR = (
    EXAMPLE_DIR / "oligomer_f_binding_balanced_monomer_only_outputs"
)

MODEL_FILES = {
    "p_p2_f": EXAMPLE_DIR / "model_p_p2_f.txt",
    "p_p2_f_binding": EXAMPLE_DIR / "model_p_p2_f_binding.txt",
    "p_p2_p4_f": EXAMPLE_DIR / "model_p_p2_p4_f.txt",
    "p_p2_p4_f_binding": EXAMPLE_DIR / "model_p_p2_p4_f_binding.txt",
    "monomer_consuming_p_p2_f": (
        EXAMPLE_DIR / "model_monomer_consuming_p_p2_f.txt"
    ),
    "monomer_consuming_p_p2_f_binding": (
        EXAMPLE_DIR / "model_monomer_consuming_p_p2_f_binding.txt"
    ),
    "monomer_consuming_p_p2_p4_f": (
        EXAMPLE_DIR / "model_monomer_consuming_p_p2_p4_f.txt"
    ),
    "monomer_consuming_p_p2_p4_f_binding": (
        EXAMPLE_DIR / "model_monomer_consuming_p_p2_p4_f_binding.txt"
    ),
}

RATE_STARTS = (0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: Path
    time_column: str
    signal_column: str
    rename_columns: dict[str, str] | None = None
    drop_local_outliers: bool = False


DATASETS = (
    DatasetConfig(
        name="example_nmr_amide_normalized",
        path=REPO_ROOT / "examples/configs/example_nmr_amide_normalized.csv",
        time_column="time",
        signal_column="amide_norm",
    ),
    DatasetConfig(
        name="real_amide_integral_timecourse_raw",
        path=EXAMPLE_DIR / "example_data.csv",
        time_column="time",
        signal_column="amide_percent",
        rename_columns={
            "Elapsed_Hours": "time",
            "Normalized_Integral (%)": "amide_percent",
            "Integral": "amide_integral",
        },
    ),
    DatasetConfig(
        name="real_amide_integral_timecourse_outlier_removed",
        path=EXAMPLE_DIR / "example_data.csv",
        time_column="time",
        signal_column="amide_percent",
        rename_columns={
            "Elapsed_Hours": "time",
            "Normalized_Integral (%)": "amide_percent",
            "Integral": "amide_integral",
        },
        drop_local_outliers=True,
    ),
)


def drop_local_signal_outliers(
    dataframe: pd.DataFrame,
    signal_column: str,
    *,
    window: int = 7,
    n_sigmas: float = 6.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = dataframe[signal_column].astype(float)
    rolling_median = values.rolling(
        window=window,
        center=True,
        min_periods=3,
    ).median()
    deviation = (values - rolling_median).abs()
    rolling_mad = deviation.rolling(
        window=window,
        center=True,
        min_periods=3,
    ).median()

    dynamic_range = float(values.max() - values.min())
    noise_floor = max(dynamic_range * 0.03, 1e-12)
    threshold = n_sigmas * rolling_mad.fillna(noise_floor).clip(lower=noise_floor)
    keep_mask = (deviation <= threshold) | rolling_median.isna()

    return (
        dataframe.loc[keep_mask].reset_index(drop=True),
        dataframe.loc[~keep_mask].reset_index(drop=True),
    )


def load_dataset(config: DatasetConfig) -> tuple[Dataset, pd.DataFrame]:
    dataframe = pd.read_csv(config.path)

    if config.rename_columns:
        dataframe = dataframe.rename(columns=config.rename_columns)

    dropped = pd.DataFrame(columns=dataframe.columns)

    if config.drop_local_outliers:
        dataframe, dropped = drop_local_signal_outliers(
            dataframe=dataframe,
            signal_column=config.signal_column,
        )

    return (
        Dataset(
            raw_dataframe=dataframe,
            time_column=config.time_column,
            signal_columns=[config.signal_column],
        ),
        dropped,
    )


def make_parameter_specs(
    model: ModelSpec,
    initial_guess: float,
) -> list[ParameterSpec]:
    return [
        ParameterSpec(
            name=parameter_name,
            initial_guess=initial_guess,
            lower_bound=0.0,
            upper_bound=100.0,
        )
        for parameter_name in model.parameters
    ]


def make_initial_condition_specs(model: ModelSpec) -> list[InitialConditionSpec]:
    specs = []

    for species_name in model.species:
        if species_name == "P":
            specs.append(
                InitialConditionSpec(
                    species="P",
                    initial_guess=1.0,
                    lower_bound=0.0,
                    upper_bound=10.0,
                    fixed=True,
                    fixed_value=1.0,
                )
            )
        else:
            specs.append(
                InitialConditionSpec(
                    species=species_name,
                    initial_guess=0.0,
                    lower_bound=0.0,
                    upper_bound=10.0,
                    fixed=True,
                    fixed_value=0.0,
                )
            )

    return specs


def make_observable_specs(dataset: Dataset, signal_column: str) -> list[ObservableSpec]:
    values = dataset.raw_dataframe[signal_column].astype(float)
    first_signal = float(values.iloc[0])
    y_max = float(values.max())
    max_abs = max(abs(first_signal), abs(y_max), 1.0)

    return [
        ObservableSpec(
            data_column=signal_column,
            species="P",
            scale_initial_guess=first_signal,
            scale_lower_bound=0.0,
            scale_upper_bound=max_abs * 5.0,
            scale_fixed=True,
            scale_fixed_value=first_signal,
            offset_initial_guess=0.0,
            offset_lower_bound=0.0,
            offset_upper_bound=max_abs * 2.0,
            offset_fixed=True,
            offset_fixed_value=0.0,
        )
    ]


def fit_one_start(
    model: ModelSpec,
    dataset: Dataset,
    signal_column: str,
    rate_initial_guess: float,
) -> FitResult:
    parameter_specs = make_parameter_specs(
        model=model,
        initial_guess=rate_initial_guess,
    )
    initial_condition_specs = make_initial_condition_specs(model)
    observable_specs = make_observable_specs(dataset, signal_column)

    settings = FitSettings(
        species_mapping={},
        rtol=1e-8,
        atol=1e-10,
        max_nfev=5000,
    )

    return fit_model(
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        observable_specs=observable_specs,
        settings=settings,
    )


def fit_model_multistart(
    model: ModelSpec,
    dataset: Dataset,
    signal_column: str,
) -> tuple[FitResult, float]:
    best_result: FitResult | None = None
    best_start = RATE_STARTS[0]

    for rate_initial_guess in RATE_STARTS:
        result = fit_one_start(
            model=model,
            dataset=dataset,
            signal_column=signal_column,
            rate_initial_guess=rate_initial_guess,
        )

        if best_result is None:
            best_result = result
            best_start = rate_initial_guess
            continue

        if result.statistics["rss"] < best_result.statistics["rss"]:
            best_result = result
            best_start = rate_initial_guess

    if best_result is None:
        raise RuntimeError("No fit attempts were run")

    return best_result, best_start


def export_result(
    result: FitResult,
    model: ModelSpec,
    dataset: Dataset,
    dataset_config: DatasetConfig,
    model_name: str,
    output_dir: Path,
    best_start: float,
) -> None:
    parameter_specs = make_parameter_specs(
        model=model,
        initial_guess=best_start,
    )
    initial_condition_specs = make_initial_condition_specs(model)
    observable_specs = make_observable_specs(dataset, dataset_config.signal_column)

    export_fit_bundle(
        fit_result=result,
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        observable_specs=observable_specs,
        species_mapping={dataset_config.signal_column: "P"},
        include_plots=True,
    )

    metadata = pd.DataFrame(
        [
            {
                "dataset": dataset_config.name,
                "model": model_name,
                "best_rate_initial_guess": best_start,
                "rss": result.statistics["rss"],
                "rmse": result.statistics["rmse"],
                "aic": result.statistics["aic"],
                "bic": result.statistics["bic"],
                "success": result.success,
                "message": result.message,
            }
        ]
    )
    metadata.to_csv(output_dir / "oligomer_fit_metadata.csv", index=False)


def fit_dataset(config: DatasetConfig) -> pd.DataFrame:
    dataset, dropped_rows = load_dataset(config)
    dataset_output_dir = BALANCED_MONOMER_ONLY_OUTPUT_DIR / config.name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    dropped_rows.to_csv(dataset_output_dir / "dropped_outliers.csv", index=False)

    fit_results: dict[str, FitResult] = {}
    rows = []

    for model_name, model_path in MODEL_FILES.items():
        model_text = model_path.read_text().strip()
        model = build_model_spec(model_text, name=model_name)
        print(f"\nDataset: {config.name}")
        print(f"Model: {model_name}")
        print(model_text)

        result, best_start = fit_model_multistart(
            model=model,
            dataset=dataset,
            signal_column=config.signal_column,
        )
        fit_results[model_name] = result

        model_output_dir = dataset_output_dir / model_name
        export_result(
            result=result,
            model=model,
            dataset=dataset,
            dataset_config=config,
            model_name=model_name,
            output_dir=model_output_dir,
            best_start=best_start,
        )

        rows.append(
            {
                "dataset": config.name,
                "model": model_name,
                "model_text": model_text.replace("\n", "; "),
                "best_rate_initial_guess": best_start,
                "success": result.success,
                "rss": result.statistics["rss"],
                "rmse": result.statistics["rmse"],
                "aic": result.statistics["aic"],
                "bic": result.statistics["bic"],
                "nfev": result.nfev,
                "message": result.message,
                **{
                    f"fit_{name}": value
                    for name, value in result.fitted_parameters.items()
                },
            }
        )

        print("Success:", result.success)
        print("Best rate initial guess:", best_start)
        print("RSS:", result.statistics["rss"])
        print("RMSE:", result.statistics["rmse"])
        print("AIC:", result.statistics["aic"])
        print("BIC:", result.statistics["bic"])
        print("Fitted parameters:", result.fitted_parameters)
        print("Fitted observable:", result.fitted_observables)

    comparison_table = build_ranked_model_comparison_table(
        fit_results=fit_results,
        sort_by="aic",
    )
    comparison_table.insert(0, "dataset", config.name)
    comparison_table.to_csv(dataset_output_dir / "model_comparison.csv", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(dataset_output_dir / "fit_summary.csv", index=False)

    return comparison_table


def main() -> None:
    BALANCED_MONOMER_ONLY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comparisons = []
    for config in DATASETS:
        comparisons.append(fit_dataset(config))

    combined = pd.concat(comparisons, ignore_index=True)
    combined.to_csv(
        BALANCED_MONOMER_ONLY_OUTPUT_DIR / "combined_model_comparison.csv",
        index=False,
    )

    print("\nCombined model comparison:")
    print(combined)
    print(f"\nWrote outputs to: {BALANCED_MONOMER_ONLY_OUTPUT_DIR}")


if __name__ == "__main__":
    main()

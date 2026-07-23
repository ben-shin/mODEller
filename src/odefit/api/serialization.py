from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): _json_safe_value(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]

    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]

    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None

    return value


def dataframe_preview(
    dataframe: pd.DataFrame | None,
    *,
    max_rows: int = 20,
) -> dict[str, Any] | None:
    if dataframe is None:
        return None

    preview = dataframe.head(max_rows).copy()

    return {
        "columns": list(preview.columns),
        "n_rows": int(len(dataframe)),
        "n_preview_rows": int(len(preview)),
        "records": [
            {
                str(key): _json_safe_value(value)
                for key, value in row.items()
            }
            for row in preview.to_dict(orient="records")
        ],
    }


def table_payload(
    dataframe: pd.DataFrame | None,
) -> dict[str, Any] | None:
    if dataframe is None:
        return None

    return {
        "columns": list(dataframe.columns),
        "n_rows": int(len(dataframe)),
        "records": [
            {
                str(key): _json_safe_value(value)
                for key, value in row.items()
            }
            for row in dataframe.to_dict(orient="records")
        ],
    }


def dataset_payload(dataset: Any, *, max_rows: int = 20) -> dict[str, Any]:
    return {
        "time_column": dataset.time_column,
        "signal_columns": list(dataset.signal_columns),
        "n_signal_columns": int(len(dataset.signal_columns)),
        "data_preview": dataframe_preview(
            dataset.raw_dataframe,
            max_rows=max_rows,
        ),
    }


def filtering_result_payload(filtering_result: Any) -> dict[str, Any]:
    if filtering_result is None:
        return {
            "kept_columns": [],
            "removed_columns": [],
            "n_kept_columns": 0,
            "n_removed_columns": 0,
        }

    return {
        "kept_columns": list(getattr(filtering_result, "kept_columns", [])),
        "removed_columns": list(getattr(filtering_result, "removed_columns", [])),
        "n_kept_columns": int(len(getattr(filtering_result, "kept_columns", []))),
        "n_removed_columns": int(len(getattr(filtering_result, "removed_columns", []))),
    }


def fit_result_payload(result: Any, *, max_rows: int = 20) -> dict[str, Any]:
    return {
        "success": bool(getattr(result, "success", False)),
        "message": str(getattr(result, "message", "")),
        "fitted_parameters": _json_safe_value(
            getattr(result, "fitted_parameters", {})
        ),
        "fitted_initial_conditions": _json_safe_value(
            getattr(result, "fitted_initial_conditions", {})
        ),
        "statistics": _json_safe_value(getattr(result, "statistics", {})),
        "simulation": dataframe_preview(
            getattr(result, "simulation_dataframe", None),
            max_rows=max_rows,
        ),
        "predicted": dataframe_preview(
            getattr(result, "predicted_dataframe", None),
            max_rows=max_rows,
        ),
        "residuals": dataframe_preview(
            getattr(result, "residuals_dataframe", None),
            max_rows=max_rows,
        ),
        "observable_table": table_payload(
            getattr(result, "observable_table", None),
        ),
    }


def model_comparison_payload(result: Any, *, max_rows: int = 20) -> dict[str, Any]:
    best_fit = getattr(result, "best_fit_result", None)

    if best_fit is None:
        best_fit = getattr(result, "best_result", None)

    return {
        "best_model_name": getattr(result, "best_model_name", None),
        "comparison_table": table_payload(
            getattr(result, "comparison_table", None),
        ),
        "best_fit": (
            fit_result_payload(best_fit, max_rows=max_rows)
            if best_fit is not None and hasattr(best_fit, "fitted_parameters")
            else None
        ),
        "n_failures": int(len(getattr(result, "failures", []))),
        "failures": [
            _json_safe_value(getattr(failure, "__dict__", failure))
            for failure in getattr(result, "failures", [])
        ],
    }


def multistart_payload(result: Any, *, max_rows: int = 20) -> dict[str, Any]:
    best_result = getattr(result, "best_result", None)

    return {
        "best_index": getattr(result, "best_index", None),
        "best_fit": (
            fit_result_payload(best_result, max_rows=max_rows)
            if best_result is not None
            else None
        ),
        "comparison_table": table_payload(
            getattr(result, "comparison_table", None),
        ),
        "n_successful_starts": int(
            len(getattr(result, "successful_results", []))
        ),
        "n_failed_starts": int(len(getattr(result, "failures", []))),
        "failures": [
            _json_safe_value(getattr(failure, "__dict__", failure))
            for failure in getattr(result, "failures", [])
        ],
    }


def bootstrap_payload(result: Any, *, max_rows: int = 20) -> dict[str, Any]:
    return {
        "original_fit": fit_result_payload(
            getattr(result, "original_result", None),
            max_rows=max_rows,
        ),
        "parameter_samples": dataframe_preview(
            getattr(result, "parameter_samples", None),
            max_rows=max_rows,
        ),
        "summary_table": table_payload(
            getattr(result, "summary_table", None),
        ),
        "n_successful_bootstrap": int(
            len(getattr(result, "bootstrap_results", []))
        ),
        "n_failed_bootstrap": int(len(getattr(result, "failures", []))),
        "failures": [
            _json_safe_value(getattr(failure, "__dict__", failure))
            for failure in getattr(result, "failures", [])
        ],
    }


def profile_likelihood_payload(result: Any, *, max_rows: int = 20) -> dict[str, Any]:
    return {
        "original_fit": fit_result_payload(
            getattr(result, "original_result", None),
            max_rows=max_rows,
        ),
        "profile_table": table_payload(
            getattr(result, "profile_table", None),
        ),
        "n_profile_points": int(
            len(getattr(result, "profile_table", []))
        ),
        "n_failures": int(len(getattr(result, "failures", []))),
        "failures": [
            _json_safe_value(getattr(failure, "__dict__", failure))
            for failure in getattr(result, "failures", [])
        ],
    }


def backend_output_payload(
    output: dict[str, Any],
    *,
    workflow: str,
    max_rows: int = 20,
) -> dict[str, Any]:
    result = output["result"]

    if workflow in {"fit", "amyloid_aggregation"}:
        result_payload = fit_result_payload(result, max_rows=max_rows)
    elif workflow == "model_comparison":
        result_payload = model_comparison_payload(result, max_rows=max_rows)
    elif workflow == "multistart":
        result_payload = multistart_payload(result, max_rows=max_rows)
    elif workflow == "bootstrap":
        result_payload = bootstrap_payload(result, max_rows=max_rows)
    elif workflow == "profile_likelihood":
        result_payload = profile_likelihood_payload(result, max_rows=max_rows)
    else:
        raise ValueError(f"Unknown workflow type: {workflow}")

    payload = {
        "workflow": workflow,
        "result": result_payload,
        "dataset": dataset_payload(
            output["dataset"],
            max_rows=max_rows,
        )
        if "dataset" in output
        else None,
        "filtering": filtering_result_payload(
            output.get("filtering_result"),
        ),
    }

    if "conditions" in output:
        payload["conditions"] = _json_safe_value(output["conditions"])

    if "written_files" in output:
        payload["written_files"] = _json_safe_value(output["written_files"])

    return payload

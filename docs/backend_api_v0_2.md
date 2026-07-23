# mODEler Backend API v0.2

This document describes the GUI-facing backend API for mODEler. The purpose of this layer is to give the GUI a small, stable interface over the modelling backend without requiring GUI code to know about parser internals, fitting internals, variable projection internals, plotting internals, or export details.

The GUI should treat these functions as the main backend contract.

## Import path

```python
from odefit.api.backend import (
    parse_model_text,
    simulate_from_text,
    fit_amyloid_aggregation_from_config,
    fit_global_observables_from_config,
    compare_global_observables_from_config,
    bootstrap_global_observables_from_config,
    profile_likelihood_global_observables_from_config,
)

from odefit.api.serialization import backend_output_payload
```

## Core design

The backend API accepts either a Python dictionary config or a path to a JSON config file.

```python
output = fit_global_observables_from_config(config)
```

or:

```python
output = fit_global_observables_from_config(
    "examples/configs/global_hsqc_variable_projection_config.json"
)
```

The raw backend output is a dictionary containing Python objects:

```python
{
    "result": result,
    "dataset": dataset,
    "filtering_result": filtering_result,
}
```

For GUI use, convert this into a JSON-friendly payload:

```python
payload = backend_output_payload(
    output,
    workflow="fit",
    max_rows=20,
)
```

The GUI should usually consume the serialized payload, not the raw result object.

## 1. Parse model text

### Function

```python
parse_model_text(model_text: str, name: str | None = None) -> dict
```

### Purpose

Parses a reaction model and returns model name, species, parameters, parsed reactions, and generated ODE lines.

### Example

```python
from odefit.api.backend import parse_model_text

parsed = parse_model_text("A -> B", name="single_step")
```

### Example output

```python
{
    "name": "single_step",
    "species": ["A", "B"],
    "parameters": ["k1f"],
    "reactions": [...],
    "ode_lines": [
        "dA/dt = -k1f*A",
        "dB/dt = k1f*A",
    ],
}
```

### GUI usage

Use this for validating model text, showing detected species, showing detected parameters, showing generated ODEs, populating parameter tables, and populating initial condition tables.

## 2. Simulate from model text

### Function

```python
simulate_from_text(
    *,
    model_text: str,
    parameters: dict[str, float],
    initial_conditions: dict[str, float],
    timepoints: list[float],
    name: str | None = None,
) -> pandas.DataFrame
```

### Purpose

Runs a simulation directly from model text and returns a dataframe.

### Example

```python
from odefit.api.backend import simulate_from_text

simulation = simulate_from_text(
    model_text="A -> B",
    parameters={"k1f": 0.4},
    initial_conditions={"A": 1.0, "B": 0.0},
    timepoints=[0.0, 1.0, 2.0, 3.0],
)
```

### GUI usage

Use this for quick model preview, the simulation panel, validating parameter values, and plotting timecourses before fitting.

## Amyloid aggregation fitting

The built-in amyloid aggregation workflow is available through:

```python
from odefit.api.backend import fit_amyloid_aggregation_from_config

output = fit_amyloid_aggregation_from_config(
    "examples/configs/amyloid_aggregation_config.json"
)
```

Serialize it for GUI use with:

```python
payload = backend_output_payload(
    output,
    workflow="amyloid_aggregation",
    max_rows=20,
)
```

See `docs/amyloid_aggregation_workflow.md` for the config format.

## 3. Fit global observables from config

### Function

```python
fit_global_observables_from_config(config_or_path)
```

### Purpose

Fits many observable columns to one kinetic model. Supported modes are single-species variable projection and multispecies variable projection.

Single-species model:

```text
peak_i(t) = scale_i * A(t) + offset_i
```

Multispecies model:

```text
peak_i(t) = c_iA*A(t) + c_iB*B(t) + ... + offset_i
```

### Example single-species config

```python
config = {
    "model_text": "A -> B",
    "data": "examples/configs/example_hsqc_peaks.csv",
    "time_column": "time",
    "observed_species": "A",
    "parameters": {
        "k1f": {
            "initial_guess": 0.1,
            "lower_bound": 0.000001,
            "upper_bound": 10.0,
        }
    },
    "initial_conditions": {
        "A": {"value": 1.0, "mode": "fixed"},
        "B": {"value": 0.0, "mode": "fixed"},
    },
    "use_variable_projection": True,
    "fit_scale": True,
    "fit_offset": True,
}
```

### Example multispecies config

```python
config = {
    "model_text": "A -> B",
    "data": "examples/configs/example_hsqc_peaks.csv",
    "time_column": "time",
    "observed_species": ["A", "B"],
    "parameters": {
        "k1f": {
            "initial_guess": 0.1,
            "lower_bound": 0.000001,
            "upper_bound": 10.0,
        }
    },
    "initial_conditions": {
        "A": {"value": 1.0, "mode": "fixed"},
        "B": {"value": 0.0, "mode": "fixed"},
    },
    "use_multispecies_variable_projection": True,
    "fit_offset": True,
}
```

### GUI serialization

```python
from odefit.api.backend import fit_global_observables_from_config
from odefit.api.serialization import backend_output_payload

output = fit_global_observables_from_config(config)
payload = backend_output_payload(output, workflow="fit", max_rows=20)
```

## 4. Compare global observable models

### Function

```python
compare_global_observables_from_config(config_or_path)
```

### Purpose

Fits multiple candidate mechanisms to the same global observable dataset and ranks them by BIC, AIC, RMSE, or RSS.

### GUI serialization

```python
output = compare_global_observables_from_config(config)
payload = backend_output_payload(output, workflow="model_comparison", max_rows=20)
```

## 5. Bootstrap uncertainty

### Function

```python
bootstrap_global_observables_from_config(config_or_path)
```

### Purpose

Runs residual bootstrap uncertainty analysis. Supported modes are single-species variable projection bootstrap and multispecies variable projection bootstrap.

Example fields:

```python
{
    "n_bootstrap": 100,
    "n_workers": 4,
    "random_seed": 123,
    "confidence_level": 0.95,
}
```

### GUI serialization

```python
output = bootstrap_global_observables_from_config(config)
payload = backend_output_payload(output, workflow="bootstrap", max_rows=20)
```

The payload contains the original fit summary, bootstrap parameter samples preview, bootstrap summary table, successful/failed replicate counts, and failure details.

## 6. Profile likelihood

### Function

```python
profile_likelihood_global_observables_from_config(config_or_path)
```

### Purpose

Runs profile likelihood analysis by fixing one parameter over a grid and refitting the remaining parameters. Supported modes are single-species variable projection profile likelihood and multispecies variable projection profile likelihood.

Example fields:

```python
{
    "profile_parameters": ["k1f"],
    "profile_n_points": 15,
    "profile_span_factor": 10.0,
    "profile_log_space": True,
}
```

### GUI serialization

```python
output = profile_likelihood_global_observables_from_config(config)
payload = backend_output_payload(output, workflow="profile_likelihood", max_rows=20)
```

## Serialized payload structure

### Fit payload

```python
{
    "workflow": "fit",
    "result": {
        "success": True,
        "message": "...",
        "fitted_parameters": {...},
        "fitted_initial_conditions": {...},
        "statistics": {...},
        "simulation": {...},
        "predicted": {...},
        "residuals": {...},
        "observable_table": {...},
    },
    "dataset": {
        "time_column": "time",
        "signal_columns": [...],
        "n_signal_columns": 150,
        "data_preview": {...},
    },
    "filtering": {
        "kept_columns": [...],
        "removed_columns": [...],
        "n_kept_columns": 150,
        "n_removed_columns": 0,
    },
}
```

### Dataframe payload

All dataframes are serialized as:

```python
{
    "columns": ["time", "peak_1", "peak_2"],
    "n_rows": 100,
    "n_preview_rows": 20,
    "records": [
        {"time": 0.0, "peak_1": 1000.0, "peak_2": 850.0}
    ],
}
```

Large dataframes are previewed by default. The GUI can request more rows by increasing `max_rows`.

## Recommended GUI architecture

The GUI should not call low-level fitting functions directly. Use:

```python
parse_model_text(...)
simulate_from_text(...)
fit_global_observables_from_config(...)
compare_global_observables_from_config(...)
bootstrap_global_observables_from_config(...)
profile_likelihood_global_observables_from_config(...)
backend_output_payload(...)
```

The GUI should collect user inputs, build a config dictionary, call one backend API function, convert output to payload, and render payload tables and plots.

## Stability notes

Backend API v0.2 is intended to be stable enough for GUI alpha work. Internal modules may continue changing, but GUI code should remain isolated from those changes by using this API layer.

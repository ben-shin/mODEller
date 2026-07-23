# Amyloid Aggregation Workflow

This workflow implements the apo amyloid aggregation fitting script as a
backend feature.

It fits multiple total-concentration conditions with shared kinetic parameters:

```text
dP/dt  = kn * m_eff^nc + k2 * m_eff^n2 * Fm
dFm/dt = 2 * ke * m_eff * dP/dt
m_eff  = smooth_max(Mtot - Fm - cm)
signal = A_condition * (Mtot - Fm) / Mtot
```

The rate parameters can be fitted in log10 space with `log_kn`, `log_k2`, and
`log_ke`, matching the legacy script.

## CLI

```bash
python -m odefit.cli fit-amyloid-aggregation \
  --config examples/configs/amyloid_aggregation_config.json
```

Override the output folder:

```bash
python -m odefit.cli fit-amyloid-aggregation \
  --config examples/configs/amyloid_aggregation_config.json \
  --output-dir examples/configs/outputs/my_amyloid_fit
```

## Python API

```python
from odefit.api.backend import fit_amyloid_aggregation_from_config
from odefit.api.serialization import backend_output_payload

output = fit_amyloid_aggregation_from_config(
    "examples/configs/amyloid_aggregation_config.json"
)
payload = backend_output_payload(output, workflow="amyloid_aggregation")
```

## File-backed Conditions

For whitespace text files like the old script, use column indices:

```json
{
  "name": "400uM",
  "data": "./apo/400uM.txt",
  "mtot": 0.0004,
  "signal_column": 0,
  "time_column": 1,
  "has_header": false,
  "delimiter": "whitespace",
  "signal_scale": 0.3,
  "amplitude_parameter": "A400",
  "fit": false
}
```

For CSV files with headers, use column names:

```json
{
  "name": "200uM",
  "data": "./apo/200uM.csv",
  "mtot": 0.0002,
  "time_column": "time",
  "signal_column": "intensity",
  "amplitude_parameter": "A200",
  "fit": true
}
```

Set `"fit": false` for conditions you want plotted/exported but excluded from
the residual vector.

## Outputs

The exporter writes:

```text
fitted_parameters.csv
fit_statistics.csv
conditions.csv
simulated_curves.csv
predicted_curves.csv
residuals.csv
summary.json
```

Supported residual modes:

```text
log
linear
relative
```

# CLI usage guide

mODEler provides command-line tools for generating ODEs, simulating models, fitting models, and running multistart fitting.

All commands are run from the repository root.

Activate the environment first:

```bash
conda activate modeler

# Generate ODEs

Input model file:

```text
A>B

run:
```bash
python -m odefit.cli generate-odes \
  --model examples/configs/model_first_order.txt \
  --output examples/configs/outputs/generated_odes.txt

This prints detected species, parameters, and generated ODEs.

For A>B, the generated system is equivalent to:
```text
dA/dt = -k1f*A
dB/dt = k1f*A

# Simulate a model
Run from CLI args:

```bash
python -m odefit.cli simulate \
  --model examples/configs/model_first_order.txt \
  --parameter-value k1f:0.5 \
  --initial-value A:1.0 \
  --initial-value B:0.0 \
  --time-start 0 \
  --time-end 10 \
  --num-points 101 \
  --output-csv examples/configs/outputs/simulation/simulation.csv \
  --output-plot examples/configs/outputs/simulation/simulation.png

Run from config:
```bash
python -m odefit.cli simulate \
  --config examples/configs/simulation_config.json

Config example:
```text
{
  "model": "examples/configs/model_first_order.txt",
  "parameter_values": {
    "k1f": 0.5
  },
  "initial_values": {
    "A": 1.0,
    "B": 0.0
  },
  "time_start": 0.0,
  "time_end": 10.0,
  "num_points": 101,
  "method": "LSODA",
  "rtol": 1e-6,
  "atol": 1e-9,
  "output_csv": "examples/configs/outputs/simulation/simulation.csv",
  "output_plot": "examples/configs/outputs/simulation/simulation.png"
}

# Model fitting
Run from CLI args:

```bash
python -m odefit.cli fit \
  --model examples/configs/model_first_order.txt \
  --data examples/configs/example_timecourse.csv \
  --time-column time \
  --signal-columns A B \
  --mapping A:A \
  --mapping B:B \
  --parameter k1f:0.2:0.001:10 \
  --initial A:1.0:fixed:0:10 \
  --initial B:0.0:fixed:0:10 \
  --output-dir examples/configs/outputs/fit \
  --no-plots

Run from config:
```bash
python -m odefit.cli fit \
  --config examples/configs/fit_config.json

Config example:
```text
{
  "model": "examples/configs/model_first_order.txt",
  "data": "examples/configs/example_timecourse.csv",
  "time_column": "time",
  "signal_columns": ["A", "B"],
  "mapping": {
    "A": "A",
    "B": "B"
  },
  "parameters": {
    "k1f": {
      "initial_guess": 0.2,
      "lower_bound": 0.001,
      "upper_bound": 10.0
    }
  },
  "initial_conditions": {
    "A": {
      "value": 1.0,
      "mode": "fixed",
      "lower_bound": 0.0,
      "upper_bound": 10.0
    },
    "B": {
      "value": 0.0,
      "mode": "fixed",
      "lower_bound": 0.0,
      "upper_bound": 10.0
    }
  },
  "output_dir": "examples/configs/outputs/fit",
  "no_plots": true
}

Output folder has fitted params, init conditions, stats, diagnostics, residuals, simulated curves, model text, and generated ODEs.

# Amyloid aggregation fitting

Run the built-in amyloid aggregation workflow from a JSON config:

```bash
python -m odefit.cli fit-amyloid-aggregation \
  --config examples/configs/amyloid_aggregation_config.json
```

This supports multiple total-concentration conditions, shared kinetic
parameters, optional per-condition amplitudes, LSODA integration, and log
residuals. See `docs/amyloid_aggregation_workflow.md`.

# Run multistart fitting
Multistart fitting runs the same model fit from different starting param guesses.

Serial run:
```bash
python -m odefit.cli multistart \
  --config examples/configs/multistart_config.json

Parallel run:
```bash
python -m odefit.cli multistart \
  --config examples/configs/multistart_config.json \
  --n-workers 4 \
  --n-starts 12

Output folder contains:
```text
multistart_comparison.csv
multistart_starting_parameters.csv
best_fit/

Parallel jobs also write:
```text
parallel_multistart_comparison.csv
parallel_multistart_failures.csv

# EXAMPLE: NMR AMIDE NORMALIZED
Simplest NMR-style CLI example:
```text
time,amide_norm
0,1.0000
1,0.9820
2,0.9700
...

Run:
```bash
python -m odefit.cli fit \
  --config examples/configs/nmr_amide_fit_config.json

This maps:
```text
amide_norm -> A

using:
```text
A>B

This is a simple direct mapping example; for real NMR data, a better model is
```text
amide_percent = scale * A + offset

Observable aware fitting is supported in the code. Will need to work on it with CLI config

# Supporteed solver methods
Simulations support scipy solve_ivp methods:
```text
RK45
RK23
DOP853
Radau
BDF
LSODA

For stiff systems, try BDF or Radau

# Current limitations
The CLI supports direct data column to species mapping, but observable mapping is only supported in the Python API. 

Custom rate laws aren't supported yet.

### Multistart global observable fitting

Run:

```bash
python -m odefit.cli multistart-global-observables \
  --config examples/configs/global_hsqc_multistart_config.json

For local parallel execution:
```bash
python -m odefit.cli multistart-global-observables \
  --config examples/configs/global_hsqc_multistart_config.json \
  --n-workers 4 \
  --n-starts 20

# Usage and Configuration

## Install and run

Use uv, Python 3.12, CMake 3.21+ and a C++20 compiler. Installing the package
builds its pybind11 extension in Release mode.

```sh
uv sync --locked
uv run --locked pytest -q
uv run --locked dmm generate --config configs/dataset_smoke.yaml
uv run --locked dmm bench --config configs/benchmark_smoke.yaml
uv run --locked dmm analyze --input results/benchmark_smoke.csv
```

Smoke produces 24 measurements: two sizes, two groups, three execution settings,
and two repeats, with one untimed warmup per case. It checks the pipeline;
its tiny timings are unsuitable for performance conclusions. A two-worker
setting requires at least two logical CPUs.

For regular runs, generate with `configs/dataset.yaml`, then benchmark with
`configs/benchmark_lite.yaml` or `configs/benchmark.yaml`. Lite covers five sizes
through 256, three groups and three execution settings: 45 measurements with two
warmups. Full covers eight sizes through 2048 with fifteen repeats and five
warmups: 1080 measurements. Scalar multiplication is cubic, so large runs can
take substantially longer.

After changing C++ code, rebuild the installed extension:

```sh
uv sync --locked --reinstall-package dmm
```

Standalone CMake builds do not replace the extension used by Python.

## Dataset settings

The default `configs/dataset.yaml` contains:

```yaml
data_dir: ../data
seed: 5206
groups: [0, 1, 2]
sizes: [16, 32, 64, 128, 256, 512, 1024, 2048]
value_range: [0, 255]
```

All paths are relative to their YAML file. Seeds and group IDs must be
non-negative integers; sizes must be positive integers. Selections must be
non-empty and unique. Bounds must be finite, representable in FP32, and ordered.
Constant bounds are supported.

For each pair, `default_rng([seed, group_id, size])` generates A and then B
uniformly within the configured bounds, rounded to FP32. NumPy multiplies the
stored inputs after conversion to FP64 to create the reference.

```text
data/
  manifest.json
  G0/
    A_16.npy
    B_16.npy
    C_ref_16.npy
    ...
  G1/
  G2/
```

The manifest records the seed, NumPy version, groups, sizes, value range,
`dtype: float32`, `reference_dtype: float64` and
`reference: numpy.matmul(float64)`. It is written last. Loads disable pickle
and check selections, shape, dtype, finiteness and input value range.
Old integer or raw binary datasets must be regenerated.

## Benchmark settings

```yaml
data_dir: ../data
output: ../results/custom.csv
cases:
  - {layout: cannon, P: 16, c: 1}
  - {layout: cannon, P: 64, c: 1}
modes:
  sequential: {}
  threaded:
    workers: [1, 2]
groups: [0, 1, 2]
sizes: [16, 32]
repeats: 15
warmups: 5
```

`repeats` defaults to fifteen positive measurements; `warmups` defaults to five
and may be zero. Sizes must be ascending and present in the dataset. Cannon
requires square P=q*q and c=1; original dimensions are padded to a multiple of P
and cropped on output. Each case requires
exactly layout/P/c, with no duplicates. Layout names are `ring`, `cannon`, and
`2.5d`. See the [layout constraints](algorithms/overview.md). Old global algorithms/P/c keys must be
migrated to cases. All cases run across the shared modes, groups and sizes. Unsupported selections fail
before output creation or worker startup.

Sequential mode takes no options. Threaded mode requires a non-empty, unique
list of worker counts between one and `min(6, os.cpu_count() or 1)`. A real
C++ thread pool is used even with one worker. Mode and worker-list ordering
is preserved. See [measurement rules](experiments.md) for loop and timing details.

## Outputs and recovery

Each run writes a CSV and a companion JSON with the same stem. CSV columns are:

```text
layout,P,c,mode,workers,group,pair,size,repeat,time_s,max_abs_error,frobenius_relative_error
```

Times are seconds. Sequential `workers` is empty. `pair` identifies the size's
index in the dataset manifest. JSON records settings, dataset metadata, platform,
CPU, package versions, Git state, native compiler/build type and timer details.

Analysis creates `<csv_stem>_analysis/` containing `summary.csv`, `time.png` and
`speedup.png`. Use `dmm analyze --input PATH --output-dir NEW_DIRECTORY` to choose
another destination.

Data, result files and analysis directories are never silently replaced.
Generation and analysis may leave partial directories after failure. Benchmark
failure preserves already validated CSV rows and run metadata. There is no
resume mechanism: inspect incomplete outputs, then use a fresh path for retries.

## Python API examples

```python
from dmm.algorithms.cannon import matmul
from dmm.execution import SerialExecutor, ThreadExecutor
import numpy as np

A = np.eye(4, dtype=np.float32)
B = np.ones((4, 4), dtype=np.float32)
with SerialExecutor() as executor:
    C = matmul(A, B, executor, P=16, c=1)
with ThreadExecutor(workers=2) as executor:
    D = matmul(A, B, executor)
np.testing.assert_array_equal(C, D)
```

`matmul_traced` returns the same product together with its BSP trace:

```python
from dmm.algorithms.cannon import matmul_traced
from dmm.bsp import bsp_cost, count

with SerialExecutor() as executor:
    C, trace = matmul_traced(A, B, executor)
counts = count(trace)  # count(trace.supersteps[s]) gives one superstep.
cost = bsp_cost(counts, w=1, g=10, l=100)
```

`bsp_cost` takes keyword-only parameters and can be recomputed for any $g$ and
$l$ without rerunning. See the [communication model](design.md#communication-model).

Import `dmm` before NumPy so the package's numerical-library thread limits apply
before libraries initialize. Direct calls reject other dtypes instead of
converting them. Read-only arrays, transposes and negative strides are supported.
The result owns independent storage. A closed thread executor cannot be reused.

See the generated [Python reference](api/index.md) for dataset and runner APIs.

Traces expose `original_size`, `padded_size`, and message `region` rectangles.
`count_native(trace)` reports the same execution excluding only input/output
conversion. Native metrics come from this same padded execution; there is no
separate native-input adapter. The common `dmm.matmul.prepare` also accepts an
explicit layout, P and c and returns the shared workspace.

Timing summaries expose median_time_s and iqr_time_s, with linearly interpolated
sample quartiles. Repetitions are summarized separately per input group; multiple
groups aggregate their medians and IQRs by median. Speedup uses matching serial
medians. Smoke/lite configurations explicitly use fewer repetitions for pipeline
checks; those settings are not the proposal performance measurement profile.

## Trace and model commands

`dmm model --config configs/model_smoke.yaml` captures real traces and produces
normalized BSP costs and budget-selection charts. `--traces PATH` recomputes
from a complete capture without loading matrices or executing algorithms.
See the [model guide](model.md) for the frozen full profile and remaining gates.


For allocation-free row-to-row timing, independent calibration and held-out
prediction checks, use `uv run dmm measure --config configs/measure_serial.yaml`
or `configs/measure_threaded.yaml`. See [measurement details](measurement.md).

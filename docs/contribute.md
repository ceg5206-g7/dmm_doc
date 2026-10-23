# Contributing an algorithm: follow Cannon

Use the existing Cannon implementation as a worked example: start with its native
algorithm, follow the Python binding, then add a prepared execution path and run
both through the shared experiment tools. The project uses **C++20**. Computation
belongs in C++; Python owns datasets, FP64 validation and experiment orchestration.

A complete contribution provides ordinary multiplication, an executed trace and
an allocation-free prepared adapter. A function that only returns the right C
cannot yet participate in all model and measurement experiments.

## 1. Find the corresponding Cannon files

Paths in this guide are relative to the main repository, including `doc/`.

| Cannon example | Responsibility when adding an algorithm |
| --- | --- |
| `include/dmm/cannon.hpp` | Declare the algorithm and check `TracedAlgorithm` |
| `src/cannon.cpp` | Implement the common input/output route and native schedule |
| `include/dmm/prepared.hpp`, `src/prepared.cpp` | Read `PreparedCannon` as the prepared-adapter example; put a new algorithm's adapter in its own header/source |
| `src/bindings.cpp` | Register NumPy entry points and the prepared object |
| `python/dmm/algorithms/cannon.py` | Expose the four Python functions used by the runners |
| `CMakeLists.txt` | Add native sources to `dmm_core` and tests to `dmm_tests` |
| `tests/cpp/test_cannon.cpp`, `tests/cpp/test_prepared.cpp` | Check arithmetic, routes, counts, coverage and allocation-free execution |
| `tests/python/test_pipeline.py`, `tests/python/test_bsp.py`, `tests/python/test_prepared.py` | Check the binding and repeated prepared calls |
| `configs/`, `doc/docs/algorithms/cannon.md` | Add experiment profiles and explain the algorithm |

For ring, create `ring.hpp`, `ring.cpp` and `python/dmm/algorithms/ring.py`;
for 2.5D use `two_point_five_d.hpp`, `two_point_five_d.cpp` and
`python/dmm/algorithms/two_point_five_d.py`. Give each its own native/Python tests
and algorithm documentation. The `ring` and `2.5d` names are already registered in
`LAYOUT_MODULES` in `python/dmm/config.py`; they are still unimplemented. The CLI
imports these modules dynamically, so adding them does not require a CLI branch.

## 2. Declare the native contract

Cannon carries the case's P/c as configuration and borrows the supplied executor:

```cpp
struct Cannon {
    std::size_t P = 16;
    std::size_t c = 1;
    Matrix<float> matmul(const Matrix<float>& a, const Matrix<float>& b,
                         Executor& executor) const;
    Traced run(const Matrix<float>& a, const Matrix<float>& b,
               Executor& executor) const;
};
static_assert(TracedAlgorithm<Cannon>);
```

This is the required contract, excerpted from `include/dmm/cannon.hpp`.
`matmul` returns `run(a,b,executor).product`; `run` returns `Traced{product,trace}`.
No inheritance hierarchy or new worker pool is needed. P is the logical node
count; it may exceed the physical worker count H.

Cannon additionally has `run_native`, but that test adapter is optional for new
algorithms. Obtain native-body metrics from `count_native(trace)` on the common
route: it excludes only `input_conversion` and `output_conversion`.

## 3. Build the traced algorithm from framework operations

Follow `Cannon::run` in `src/cannon.cpp` in this order:

1. Validate matching square storage, square P=q² and c=1.
2. Construct `Bsp machine(P)`. `make_row_shards(machine,a)` pads to
   `N=ceil(n/P)*P` and distributes N/P rows to each logical node.
3. Call `rows_to_blocks(machine,executor,rows,1,tag)` for A and B. Release
   consumed row buffers when they are no longer needed.
4. Run the skew and q multiply steps in `multiply_blocks`. Each node accesses
   its own buffers and communicates through `send`/`receive`.
5. Call `blocks_to_rows` on C, then gather/crop the common row result.
6. Finish the trace and set original/padded sizes and the coverage partition.

Use tracked `Buffer` storage from `Bsp`/`Node` for node payload, packing and
replicas. Do not let node tasks read other nodes' arrays or reload blocks from
the full host input. Messages sent in one step must be consumed in the next;
self-owned data stays local. Matrix messages need their actual global
`ElementRegion{row,col,rows,cols}` and distinct destination/tag pairs per sender.
Payload sizes are FP32 words, not bytes.

Cannon's actual multiplication hook is:

```cpp
node.multiply_add({i, k, j, block, block, block},
                  a[id], block, b[id], block, c[id], block);
```

Here `k=(i+j+round)%q`. This calls the shared tile-32 GEMM and records successful
work/coverage together. Do not add the same GEMM FLOPs with `compute`, write a
private multiplication loop, change the shared tile or synthesize product
records after execution. Reduction additions use `compute` without products;
the shared tree-reduction helper already counts its additions.

Cannon sets `n=left.size` (the padded size) and completes its metadata with:

```cpp
trace.original_size = a.size;
trace.padded_size = n;
trace.coverage = {q, q, q};
```

Run `validate_coverage(trace)` alongside the independent FP64 correctness check.
For a row ring the partition is `{P,P,1}`; for 2.5D it is `{q,q,q}` across all
layers. Ring already uses the common row layout. 2.5D uses the shared base-layer
conversions, tree broadcasts and tree reduction back to layer 0. Do not copy
Cannon's square-grid validation into ring: ring only requires positive P and c=1.
See the [C++ API](api/cpp.md) for exact buffer, collective and conversion contracts.

## 4. Register bindings and expose Python entry points

In `src/bindings.cpp`, follow the existing `cannon`/`cannon_traced` wrappers and
`PreparedCannon` registration. They validate NumPy inputs and P/c, copy inputs into
owned C++ storage, borrow the executor, release the GIL for native execution and
return independent output. Keep these checks when adapting the wrappers:

- Matching, finite, nonempty square float32 inputs; preserve read-only/strided inputs.
- Exact integer P/c, rejecting booleans and invalid grids before execution.
- A prepared output must be writable, C-contiguous float32 with the original shape.
- Keep buffers alive until the synchronous executor call completes; serialize
  shared prepared-state reset/execution/gather as `PreparedCannon` does.

The Python module then exposes the following contract, as Cannon does:

```python
def validate_size(size: int, *, P: int, c: int) -> None: ...
def matmul(A, B, executor, *, P: int, c: int): ...
def matmul_traced(A, B, executor, *, P: int, c: int): ...
def prepare(A, B, *, P: int, c: int): ...
```

`validate_size` is a cheap preflight without matrices or workers. Cannon checks
positive size, `_native.grid_size(P,c)`, c=1 and `_native.padded_size(size,P)`.
It accepts non-divisible sizes through common padding. Direct native entry
points still validate their inputs; preflight is not a substitute.

Cannon's other three functions forward to `_native.cannon`,
`_native.cannon_traced` and `_native.PreparedCannon`. The ordinary benchmark calls
`matmul`; model capture calls `matmul_traced` and receives `(C,Trace)`; measurement
calls `prepare`, then the prepared object's `run_into` and `trace`. Keep I/O,
oracle computation, parameter fitting and report generation out of these wrappers.

## 5. Add the prepared execution path

Read `PreparedCannon` in `src/prepared.cpp`. It declares operations once on
`PreparedBsp`, then seals the plan. A new adapter uses the same public operations:
`allocate`, `step`, `zero`, `copy`, `send`, `receive` and `multiply_add`.
`BufferView{buffer,offset,stride}` identifies local storage; sends return message
ids for matching next-step receives. Shared tree collectives and row/block
conversions also have prepared overloads, including counted scratch for aliases.

The adapter exposes `run_into(output,executor)` returning `RunTimings`, and
`trace()` returning the last successful executed trace. Each run resets common
input rows outside the internal timer and gathers/crops the host output after it.
Zeroing C, packing, conversions, copies, computation and reduction remain inside
the timed plan. Even the first execution must allocate no payload or metadata;
allocation and a trace snapshot copy belong outside the timer. All allocated
node payload, including retained message caches, remains counted in memory peaks.

The fixed phase names are `input_conversion`, `skew`, `replication`, `multiply`,
`reduction`, `output_conversion`. Return the framework's end-to-end and six-phase
timing structure. Prepared and ordinary routes should agree on outputs and
work/message schedules; their storage peaks can differ and must be reported
separately. See [prepared measurement](measurement.md) for timing components.

## 6. Try the complete Cannon API

After `uv sync --locked --reinstall-package dmm`, run this Python example. It uses
17×17 inputs and P=4, so the traced computation pads to 20×20; serial execution
also exercises P>H. Both results are checked against an independently computed
FP64 reference rather than against each other alone.

```python
import numpy as np
from dmm.algorithms import cannon
from dmm.bsp import count, validate_coverage
from dmm.execution import SerialExecutor

rng = np.random.default_rng(5206)
a = rng.uniform(-1, 1, (17, 17)).astype(np.float32)
b = rng.uniform(-1, 1, (17, 17)).astype(np.float32)
reference = a.astype(np.float64) @ b.astype(np.float64)
a.flags.writeable = b.flags.writeable = False
cannon.validate_size(17, P=4, c=1)

with SerialExecutor() as executor:
    product, trace = cannon.matmul_traced(a, b, executor, P=4, c=1)
    np.testing.assert_allclose(product, reference, atol=1e-4, rtol=1e-3)
    validate_coverage(trace)
    assert trace.original_size == 17 and trace.padded_size == 20
    assert count(trace).total_flops == 2 * 20**3

    prepared = cannon.prepare(a, b, P=4, c=1)
    output = np.empty_like(a)
    for _ in range(2):
        output.fill(np.nan)
        timings = prepared.run_into(output, executor)
        np.testing.assert_allclose(output, reference, atol=1e-4, rtol=1e-3)
        validate_coverage(prepared.trace())
        assert timings.end_to_end.total > 0
```

## 7. Exercise the model and measurement runners

Create these three YAML files under `configs/`. Paths resolve relative to each
YAML, not the shell's working directory. Use new data/output paths if these
already exist; the tools refuse to overwrite existing datasets/reports.

`configs/dataset_contribute.yaml`:

```yaml
data_dir: ../data_contribute
seed: 5206
groups: [0]
sizes: [17, 65]
value_range: [-1, 1]
```

`configs/model_contribute.yaml`:

```yaml
data_dir: ../data_contribute
output_dir: ../results/model_contribute
cases:
  - {layout: cannon, P: 4, c: 1}
  - {layout: cannon, P: 16, c: 1}
sizes: [17, 65]
group: 0
workers: 0
gammas: [1, 10]
lambdas: [100]
budgets: [8, 32]
```

`configs/measure_contribute.yaml`:

```yaml
data_dir: ../data_contribute
output_dir: ../results/measure_contribute
cases:
  - {layout: cannon, P: 4, c: 1}
sizes: [17, 65]
group: 0
workers: 0
warmups: 5
repeats: 15
```

From the main checkout:

```sh
uv run --locked dmm generate --config configs/dataset_contribute.yaml
uv run --locked dmm model --config configs/model_contribute.yaml
uv run --locked dmm measure --config configs/measure_contribute.yaml
```

Model capture must produce validated traces, correctness metrics and model/budget
CSVs and figures. Measurement must produce raw samples, frozen calibration,
prepared traces, median/IQR, timing components and held-out residuals. These small
runs check integration, not performance rankings. The [model guide](model.md) and
[measurement guide](measurement.md) describe the output schemas and offline replay.

For a new algorithm, select its layout/P/c in separate profiles before running
the full proposal profile. Unimplemented layouts fail preflight; neither a name
in YAML nor a cardinality test establishes that the algorithm works.

<a id="verification"></a>

## 8. Test and submit the contribution

Adapt Cannon's tests to cover FP64 correctness, zero/identity/random/cancellation
inputs, padding/cropping (including 17,65,257), invalid configurations, input
preservation, output ownership, read-only/strided inputs and repeated calls.
Exercise serial and threaded executors, including P>H. Add hand-counted work and
message-route tests and prove each executed product triple occurs exactly once.
For Cannon, n=16/P=16 gives S=11, F_total=8192, C_total=2496 and C_crit=164 on the
common route; see its [count derivation](algorithms/cannon.md#hand-computed-communication-counts).
Use your own algorithm's expected schedule, not these Cannon constants.

Prepared tests must verify no allocations on caller/workers, including the first
run, no C accumulation across runs and correct recovery after an executor error.
Reuse/extend the allocation-counter tests in `tests/cpp/test_prepared.cpp`;
do not define another global `operator new` in a second linked test file.

Run the following from the main checkout (configure a fresh Debug directory too
when changing native memory/message handling):

```sh
uv sync --locked --reinstall-package dmm
uv run cmake --preset release
uv run cmake --build --preset release --parallel
ctest --preset release --no-tests=error
uv run --locked pytest -q tests/python doc/tests
(cd doc && uvx --with-requirements requirements.txt mkdocs build --strict)
```

Add an algorithm page under `doc/docs/algorithms/`, link it in `doc/mkdocs.yml`,
and update the [status table](algorithms/overview.md) after implementation and
verification. Record layout constraints, node numbering, schedules, counts and
measurement limits. Commit `doc/` first, then commit the parent repository's
updated submodule pointer along with source/tests/configs. Coordinate edits to
`CMakeLists.txt`, bindings and navigation with other contributors. Do not weaken
shared validation or change the shared GEMM/statistics/model to make one layout
pass. Full comparison remains gated on actual implementations of all layouts.

For diagrams, use Mermaid fences with `accTitle` and `accDescr`; use `$...$` or
`$$` math fences as in the [Cannon page](algorithms/cannon.md). Keep architecture,
[API contracts](api/cpp.md) and [measurement protocol](measurement.md) consistent.

# Adding an algorithm

Algorithm logic lives in one plan-building function. It declares owned
buffers, local operations and messages. The common `PreparedMatmul` handles
input copies, padding, reset, cropped output, locking and timing. The same
`BspPlan` interpreter serves ordinary and retained calls. Do not implement a
second prepared class or a separate ordinary multiplication loop. Integration
also needs registration, a thin Python module, tests and documentation, listed
below. The [verification record](verification.md#complete-cannon-pipeline-2026-10-23)
shows the complete pipeline already exercised with Cannon.

## 1. Files and responsibilities

| File | Responsibility |
| --- | --- |
| `include/dmm/algorithms.hpp` | Declare algorithm builder functions |
| `src/algorithms/cannon.cpp` | Allocate/bind buffers and declare Cannon's schedule |
| `src/algorithms/registry.cpp` | Select a builder by layout name; common ordinary entry points |
| `include/dmm/matmul.hpp`, `src/matmul.cpp` | Shared workspace lifecycle |
| `include/dmm/plan.hpp`, `src/plan_build.cpp`, `src/plan_execute.cpp` | DSL, validation and execution |
| `include/dmm/redistribution.hpp`, `include/dmm/collectives.hpp` | Shared conversion and tree-building helpers |
| `src/bindings.cpp`, `python/dmm/matmul.py` | Common binding/NumPy protocol |
| `python/dmm/algorithms/cannon.py` | Thin forwarding functions for runner discovery |

| Integration point | 1D ring | 2.5D |
| --- | --- | --- |
| Layout string | `ring` | `2.5d` |
| Builder | `build_ring` | `build_two_point_five_d` |
| C++ source | `src/algorithms/ring.cpp` | `src/algorithms/two_point_five_d.cpp` |
| Python forwarding module | `python/dmm/algorithms/ring.py` | `python/dmm/algorithms/two_point_five_d.py` |
| C++ tests | `tests/cpp/test_ring.cpp` | `tests/cpp/test_two_point_five_d.cpp` |
| Algorithm page | `doc/docs/algorithms/ring.md` | `doc/docs/algorithms/two_point_five_d.md` |

Declare the builder in `include/dmm/algorithms.hpp`, select it for the layout in
`src/algorithms/registry.cpp`,
and add the source and C++ tests to `CMakeLists.txt`. Add Python integration tests
under `tests/python/` and update `doc/mkdocs.yml` navigation. Layout module names
are already registered in `python/dmm/config.py`. No new switch or class is
needed in `src/bindings.cpp`: new layouts use the existing bindings, prepared
class, common NumPy conversion and output validation.

Tests follow responsibilities: `test_plan.cpp`, `test_matmul.cpp`,
`test_collectives.cpp`, `test_redistribution.cpp`, `test_trace.cpp`,
`test_validation.cpp` and one `test_<algorithm>.cpp` per algorithm.

## 2. Build the Cannon plan

The only extension interface is:

```cpp
MatmulDefinition build_cannon(BspPlan& plan, const MatmulConfig& config);
```

The builder receives `config` by const reference: `padded_size`, `P` and `c`.
Write N=`config.padded_size`; the common lifecycle already chose N=ceil(n/P)*P
from the original input size n. Return one MatrixView per
node for `input_a_rows`, `input_b_rows` and `output_c_rows`, each N/P×N with the
correct owner and distinct A/B/C buffers; return `product_grid={q,q,q}`. The
workspace fills these inputs before execution and gathers the output afterwards.
All other algorithm work is declared in the plan. Allocate buffers while
building; `add_*` registers work to execute later. Include `add_zero` for every
C accumulator before its first GEMM on every execution. The common lifecycle
restores A/B, but does not zero C for the builder.

Cannon reuses tile buffers for common rows. Each has b²=N²/P words; separate h×b
scratch preserves self segments during in-place conversion. This is the complete
implementation in `src/algorithms/cannon.cpp`:

```cpp
#include "dmm/algorithms.hpp"
#include "dmm/redistribution.hpp"

namespace dmm {
MatmulDefinition build_cannon(BspPlan& plan, const MatmulConfig& config) {
    if (config.c != 1)
        throw std::invalid_argument("Cannon requires c = 1");
    const auto N = config.padded_size, P = config.P, q = grid_size(P);
    if (!N || N % P || plan.nodes() != P)
        throw std::invalid_argument("invalid Cannon configuration");
    const auto block = N / q, height = N / P;
    std::vector<MatrixView> A(P), B(P), C(P), scratch(P);
    MatmulDefinition definition{
        std::vector<MatrixView>(P), std::vector<MatrixView>(P), std::vector<MatrixView>(P), {q, q, q}};
    auto allocate_matrix = [&](std::size_t node, std::size_t rows, std::size_t cols) {
        return plan.matrix_view(plan.allocate_buffer(node, rows * cols), rows, cols);
    };
    for (std::size_t node = 0; node < P; ++node) {
        A[node] = allocate_matrix(node, block, block);
        B[node] = allocate_matrix(node, block, block);
        C[node] = allocate_matrix(node, block, block);
        definition.input_a_rows[node] = plan.matrix_view(A[node].buffer, height, N);
        definition.input_b_rows[node] = plan.matrix_view(B[node].buffer, height, N);
        definition.output_c_rows[node] = plan.matrix_view(C[node].buffer, height, N);
        if (P > 1)
            scratch[node] = allocate_matrix(node, height, block);
    }
    if (P > 1) {
        add_rows_to_tiles(plan, config, definition.input_a_rows, A, scratch);
        add_rows_to_tiles(plan, config, definition.input_b_rows, B, scratch);
    }
    auto node_at = [q](std::size_t row, std::size_t col) { return (row % q) * q + col % q; };
    auto skew = plan.add_superstep(Phase::Skew);
    for (std::size_t node = 0; node < P; ++node) {
        const auto i = node / q, j = node % q;
        skew.add_zero(C[node]);
        if (i)
            skew.add_send(node_at(i, j + q - i), PayloadPart{A[node], {i * block, j * block}});
        if (j)
            skew.add_send(node_at(i + q - j, j), PayloadPart{B[node], {i * block, j * block}});
    }
    for (std::size_t round = 0; round < q; ++round) {
        auto step = plan.add_superstep(Phase::Multiply);
        for (std::size_t node = 0; node < P; ++node) {
            const auto i = node / q, j = node % q, k = (i + j + round) % q;
            const auto source_a = node_at(i, j + (round == 0 ? i : 1));
            const auto source_b = node_at(i + (round == 0 ? j : 1), j);
            if (source_a != node)
                step.add_receive(source_a, A[node]);
            if (source_b != node)
                step.add_receive(source_b, B[node]);
            step.add_gemm(A[node], B[node], C[node], {i, k, j});
            if (round + 1 < q) {
                step.add_send(node_at(i, j + q - 1), PayloadPart{A[node], {i * block, k * block}});
                step.add_send(node_at(i + q - 1, j), PayloadPart{B[node], {k * block, j * block}});
            }
        }
    }
    if (P > 1)
        add_tiles_to_rows(plan, config, C, definition.output_c_rows, scratch);
    return definition;
}
} // namespace dmm
```

The schedule reads in order: convert A/B, initialize C and skew, receive/GEMM/
shift for each round, convert C back, return the bindings. Source formulas remove
incoming-message tables. `add_gemm` infers dimensions and records work/coverage.
`PayloadPart{view,{row,col}}` annotates the actual global rectangle; matching uses
only the sending step and endpoints. All operations on each node follow copies,
GEMMs, sends. Never append operations to an older superstep.

There is one message per ordered node pair per step. For several components use
`step.add_send(destination, {{a, origin_a}, {b, origin_b}})` and next-step
`step.add_receive(source, {local_a, local_b})`. Components are copied and matched
positionally. Shapes/arity are checked; equal-shaped swapped operands remain the
author's responsibility. No tag or message handle is needed. See the full
[DSL contract](api/cpp.md#bsp-operation-plan).

The builder never calls `execute` or `finalize`, starts a pool, runs a GEMM or
synthesizes a trace. Ordinary and retained calls execute these same operations.

## 3. Ordinary and retained calls

```cpp
dmm::PreparedMatmul job(a, b, P, c, dmm::build_cannon);
std::vector<float> output(a.values.size());
job.run_into(output, executor);
auto trace = job.trace();
job.run_into(output, executor); // same inputs, clean C, same plan
```

The constructor owns input copies. `run_into` serializes restore/execute/gather;
C zeroing is an operation in the timed plan. Inputs and output host adapters,
allocation and trace snapshot copying are outside internal timing. Ordinary
`matmul` builds the same workspace temporarily. Complete traces and peak memory
must match the retained route. Native metrics filter conversion phases from this
same execution; there is no standalone native-input adapter to implement.

Python layout modules forward to shared functions; Cannon's pattern is:

```python
from dmm import matmul as shared

def prepare(A, B, *, P=16, c=1):
    return shared.prepare(A, B, layout="cannon", P=P, c=c)
```

Likewise expose forwarding `validate_size`, `matmul` and `matmul_traced` for the
existing runners. Register the C++ builder in `src/algorithms/registry.cpp`; do not
copy NumPy conversion, GIL handling or prepared-class registration. Inputs are
finite matching FP32 square matrices; strided/read-only inputs are preserved.
Prepared output is writable contiguous FP32 of original shape.

## 4. Ring builder

The [Ring builder](algorithms/ring.md) uses
`MatmulDefinition build_ring(BspPlan&, const MatmulConfig&)`.
Require positive P and c=1; P need not be square. Let N=config.padded_size
and h=N/P. Bind A/C h×N rows and a circulating B h×N row block per node.
For a ring sending to (r+1)%P, round t uses k=(r−t+P)%P:

$$
C_r \mathrel{+}= A_r[:,kh:(k+1)h] B_k.
$$

Call `add_gemm(A_rows[r].submatrix(0,k*h,h,h), B[r], C[r], {r,k,0})`
and return `product_grid={P,P,1}`. For t>0, call
`step.add_receive((r+P-1)%P, B[r])` before the GEMM. Except in the last round,
send `PayloadPart{B[r], {k*h,0}}` to (r+1)%P; its trace region is `{k*h,0,h,N}`.
Zero C before the first GEMM and run P rounds. Native layout is already common
rows, so no row/block conversion is needed. With one step per round, expect
F_total=2N³, W_crit=2N³/P, P(P−1) messages, C_total=(P−1)N²,
C_crit=(P−1)N²/P, S=P. Counts use padded N; reported useful GFLOP/s uses
original n. Derive/assert these independently of runtime counting.
Test P=1/2/3 and proposal P=16/64/256, including P>H.

## 5. 2.5D builder

The [2.5D builder](algorithms/two_point_five_d.md) uses
`MatmulDefinition build_two_point_five_d(BspPlan&, const MatmulConfig&)`.
Validate with `grid_size(P,c)`: P=c*q² and c divides q.
Use id(z,i,j)=z*q²+i*q+j so the first q² IDs form the base layer.
Use shared conversions, tree broadcasts of A/B over
ordered groups `[id(0,i,j),...,id(c−1,i,j)]`, then layer GEMMs, tree sum of C to
z=0 and shared conversion back to common rows. Let N=config.padded_size.
Bind common A/B/C row views for all P nodes, including nonbase layers. Allocate
every participating layer's A/B/C tile storage while building, before calling
the collective helpers. The conversion helper's tile list only needs populated
entries for the base q² nodes; this does not permit empty views for participants
in a tree's `NodePayloads`.

Build `NodePayloads` with A then B components and broadcast them together, so
each tree edge sends one message. Every participant needs owned views with
matching component shapes and global origins. Zero all C accumulators before
their first GEMM, then reduce C through `add_tree_reduce_sum`. The tree helpers
append their complete send/receive sequence; continue with a new superstep.
At c=1 singleton collectives append no steps; outputs and the declared schedule
must agree with Cannon, including common conversions and skew. Use the same
builder and lifecycle for all c.

One valid layered Cannon schedule takes R=q/c rounds per layer. Initially shift
A left by i+zR and B up by j+zR, modulo q. In round t use
k=(i+j+zR+t)%q; between rounds shift A left and B up by one. Use block side b=N/q,
`add_gemm(A[node],B[node],C[node],{i,k,j})`, and return
`product_grid={q,q,q}`. Across layers/rounds each
(i,k,j) must occur exactly once. Do not run a full Cannon on every layer.

Test P=64/c=4 (R=1), P=256/c=4 (R=2), and c=1 degeneration. Verify 2N³ GEMM
FLOPs plus (c−1)N² reduction additions, actual coverage and all replicated/cache
memory. Public helpers count reduction additions automatically. Do not force a
configuration through a memory budget when its recorded peak exceeds the cap.

## 6. Model and measurement smoke profiles

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
  - {layout: cannon, P: 1, c: 1}
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

The measurement smoke uses P=1/H=1. Model capture covers P=4/16 conversions
with batched execution. Formal `measure` runs reject P>H. For four-core parallel
measurements use P=4/H=4 and pass a completed P=1/H=1 report with `--baseline`
as described in the [measurement guide](measurement.md).

Model capture must produce validated traces, correctness metrics and model/budget
CSVs and figures. Measurement must produce raw samples, frozen calibration,
prepared traces, median/IQR, timing components and held-out residuals. These small
runs check integration, not performance rankings. The [model guide](model.md) and
[measurement guide](measurement.md) describe the output schemas and offline replay.

For a new layout, select its layout/P/c in separate profiles before running
the full proposal profile. Unimplemented layouts fail preflight; neither a name
in YAML nor a cardinality test establishes that the algorithm works.

<a id="verification"></a>

## 7. Test and submit

Use independent FP64 oracles, zero/identity/random/signed/cancellation inputs,
padding/cropping at 17/65/257, invalid grids and serial/threaded execution
including P>H. Assert actual message routes, critical/total counts and exact
block coverage. Check the complete trace (regions, products, work, phases,
communication and peaks) agrees between ordinary and repeated prepared calls.
These paths share code, so agreement alone is not a correctness oracle.

Common lifecycle/NumPy boundary tests belong to the shared suite. A new builder
must additionally pass first/repeated no-allocation and failure/retry checks.
Use `allocation_test::Watch` from `tests/cpp/allocation_watch.hpp`; the one global
allocation hook is in `test_allocations.cpp`. Do not define another `operator new`. Keep ownership, invalid-view, duplicate
message and unconsumed-message validation. Do not weaken the kernel or statistics.

```sh
uv sync --locked --reinstall-package dmm
uv run cmake --preset debug
uv run cmake --build --preset debug --parallel
uv run ctest --preset debug --no-tests=error
uv run cmake --preset release
uv run cmake --build --preset release --parallel
uv run ctest --preset release --no-tests=error
uv run --locked pytest -q tests/python doc/tests
(cd doc && uvx --with-requirements requirements.txt mkdocs build --strict)
```

Update the algorithm page, `doc/mkdocs.yml` and status overview after
verification. Use captured traces and measurements when updating the
[formal comparison](results.md), and preserve the run revision and evidence.

Commit `doc/` first, then source, tests, configs and the updated submodule
pointer in the parent repository.

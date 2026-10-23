# Adding an algorithm

An algorithm contribution is one plan-building function. It declares owned
buffers, local operations and messages. The common `PreparedMatmul` handles
input copies, padding, reset, cropped output, locking and timing. The same
`BspPlan` interpreter serves ordinary and retained calls. Do not implement a
second prepared class or a separate ordinary multiplication loop.

## 1. Files and responsibilities

Cannon is the worked example:

| File | Responsibility |
| --- | --- |
| `include/dmm/cannon.hpp` | Declare `build_cannon(BspPlan&,MatmulLayout&)`; the existing `Cannon` facade is compatibility only |
| `src/cannon.cpp` | Allocate/bind buffers and declare Cannon's schedule |
| `include/dmm/matmul.hpp`, `src/matmul.cpp` | Shared workspace lifecycle for all layouts |
| `include/dmm/bsp_plan.hpp`, `src/bsp_plan.cpp` | Single operation interpreter and plan validation |
| `src/bindings.cpp` | Shared prepare/matmul/output protocol; select the layout's builder |
| `python/dmm/matmul.py` | Common Python API |
| `python/dmm/algorithms/cannon.py` | Thin forwarding functions for runner discovery |

For ring add `include/dmm/ring.hpp`, `src/ring.cpp`, corresponding C++/Python
tests, `python/dmm/algorithms/ring.py` and `doc/docs/algorithms/ring.md`.
For 2.5D use `two_point_five_d` in those filenames. Coordinate CMake source/test
registration, the small builder switch in `src/bindings.cpp`, and documentation
navigation. Layout module names are already registered in `config.py`.
Neither new layout needs a workspace class, new bound class or its own NumPy
validation/output implementation.

## 2. Declare buffers and build the Cannon plan

The extension contract is:

```cpp
void build_cannon(BspPlan& plan, MatmulLayout& layout);
```

`layout.n` is the padded size; `layout.P` and `layout.c` are immutable case
parameters. The builder fills three lists: `a`, `b`, `result`, one owned buffer
ID per node, capacity at least n²/P. These buffers contain common row shards
at the start/end of execution. Native tiles can reuse the same capacity. The
builder also sets `layout.coverage` and declares every required C zeroing step.
The framework seals the plan after validating these bindings.

Cannon validates P=q², c=1, then allocates one b×b A/B/C tile per node, where
b=n/q. Its common row shards fit in the same buffers, so conversion uses
separate h×b scratch (h=n/P). This preserves its existing payload memory count.

A representative excerpt from `src/cannon.cpp` is:

```cpp
auto& left = layout.a;
auto& right = layout.b;
auto& result = layout.result;
for (std::size_t node = 0; node < P; ++node) {
    left.push_back(plan.allocate(node, block * block));
    right.push_back(plan.allocate(node, block * block));
    result.push_back(plan.allocate(node, block * block));
    if (P > 1) scratch.push_back(plan.allocate(node, height * block));
}
if (P > 1) {
    rows_to_blocks(plan, n, 1, left, left, 0, scratch);
    rows_to_blocks(plan, n, 1, right, right, 1, scratch);
}
```

It next declares skew sends, then q multiply steps. Each step receives pending
messages before the GEMM and sends next-round operands afterwards:

```cpp
plan.multiply_add(step, node,
                  {i, k, j, block, block, block},
                  {left[node], 0, block},
                  {right[node], 0, block},
                  {result[node], 0, block});
```

Here k=(i+j+round)%q. `multiply_add` executes the common blocked GEMM and records
work/coverage; do not manually count the same work. Omit final-round shifts.
Finish with `blocks_to_rows` and `layout.coverage={q,q,q}`. The full source
contains message routing and C zeroing; do not add host reset/gather code here.

The builder runs once during construction. It never calls `execute`, `seal`,
starts a thread pool, performs actual GEMMs, or synthesizes a trace.

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
existing runners. Register the C++ builder in the shared binding switch; do not
copy NumPy conversion, GIL handling or prepared-class registration. Inputs are
finite matching FP32 square matrices; strided/read-only inputs are preserved.
Prepared output is writable contiguous FP32 of original shape.

## 4. Ring builder

Implement `void build_ring(BspPlan&,MatmulLayout&)`. Require positive P and c=1;
P need not be square. Let h=n/P. Bind A/C h×n rows and a circulating B h×n row
block per node. For a ring sending to (r+1)%P, round t uses k=(r−t+P)%P:

$$
C_r \mathrel{+}= A_r[:,kh:(k+1)h] B_k.
$$

Record `ProductRegion{r,k,0,h,n,h}` and coverage `{P,P,1}`. B's message region is
`{k*h,0,h,n}`. Run P rounds, omitting the last send. Native layout is already
common rows, so no row/block conversion is needed. With one step per round,
expect F_total=2n³, W_crit=2n³/P, P(P−1) messages, C_total=(P−1)n²,
C_crit=(P−1)n²/P, S=P. Derive/assert these independently of runtime counting.
Test P=1/2/3 and proposal P=16/64/256, including P>H.

## 5. 2.5D builder

Implement `void build_two_point_five_d(BspPlan&,MatmulLayout&)`. Validate with
`grid_size(P,c)`: P=c*q² and c divides q. Use id(z,i,j)=z*q²+i*q+j so the first
q² IDs form the base layer. Use shared conversions, tree broadcasts of A/B over
ordered groups `[id(0,i,j),...,id(c−1,i,j)]`, then layer GEMMs, tree sum of C to
z=0 and shared conversion back to common rows. Preallocate all participants.

One valid layered Cannon schedule takes R=q/c rounds per layer. Initially shift
A left by i+zR and B up by j+zR, modulo q. In round t use
k=(i+j+zR+t)%q; between rounds shift A left and B up by one. Use block side b=n/q,
`ProductRegion{i,k,j,b,b,b}`, and coverage `{q,q,q}`. Across layers/rounds each
(i,k,j) must occur exactly once. Do not run a full Cannon on every layer.

Test P=64/c=4 (R=1), P=256/c=4 (R=2), and c=1 degeneration. Verify 2n³ GEMM
FLOPs plus (c−1)n² reduction additions, actual coverage and all replicated/cache
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
Extend the existing allocation hook in `tests/cpp/test_prepared.cpp`; do not
define another global `operator new`. Keep ownership, invalid-view, duplicate
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

Update the algorithm page, `doc/mkdocs.yml` and status overview only after
verification. Commit `doc/` first, then source/tests/configs and the updated
submodule pointer in the parent repository. Both teammate algorithms must be
implemented before running the full 24-configuration/192-scenario comparison.

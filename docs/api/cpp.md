# C++ API

## Matrix and executor

`matrix.hpp` provides owning `Matrix<T>{size,values}` and checked read-only
`MatrixBlockView<T>`. Matrices use checked square storage. FP32 products are
checked against an independent FP64 reference using `validation.hpp`.

`execution.hpp` provides synchronous `Executor::for_each(count,task)`,
`SerialExecutor`, `ThreadExecutor(workers)` and `make_executor(workers)` (zero
selects serial). Logical P is independent of H threads. Executor batches block
until all tasks finish and propagate errors. See [lifecycle](../design.md#executors).

## Algorithm builder

`matmul.hpp` defines the extension contract:

```cpp
struct MatmulConfig {
    std::size_t padded_size, P, c;
};
struct MatmulDefinition {
    std::vector<MatrixView> input_a_rows, input_b_rows, output_c_rows;
    CoverageGrid product_grid;
};
using PlanBuilder = MatmulDefinition (*)(BspPlan&, const MatmulConfig&);
```

A builder receives immutable case parameters and returns the common input/output
bindings and product partition. Each list contains exactly P views, indexed by
node, with shape `(padded_size/P, padded_size)` and matching owner. A/B/C row
bindings must use distinct buffers. Native tiles may reuse the row storage or
occupy extra capacity. The builder allocates all node storage and declares C
zeroing and operations; the workspace validates bindings and finalizes the plan.

`algorithms.hpp` declares builders, `select_builder(layout)` and the ordinary
`matmul` / `matmul_traced` entry points. Their trailing arguments are
`layout="cannon", P=16, c=1`. Only Cannon is implemented. Ring/2.5D fail explicitly.

## Preallocated execution

```cpp
PreparedMatmul(Matrix<float> a, Matrix<float> b,
               std::size_t P, std::size_t c, PlanBuilder build);
std::size_t size() const noexcept;
RunTimings run_into(std::span<float> output, Executor& executor);
Trace trace() const;
Traced run(Executor& executor); // owning product and trace snapshot
```

`PreparedMatmul` owns input copies, applies common padding, builds and finalizes
once. Every run serializes input restoration, execution and cropped gathering.
Output needs exactly original_n² words. First and repeated execution allocate
nothing. C zeroing executes inside the timed plan. Allocation, host reset/gather,
validation and snapshot copying are outside internal timing. Ordinary calls
create this same workspace temporarily.

`trace()` requires a successful execution. Failure invalidates it; a subsequent
successful run recovers the workspace. `run()` keeps execution and its snapshot
under the same workspace lock.

```cpp
dmm::PreparedMatmul job(a, b, 16, 1, dmm::build_cannon);
std::vector<float> output(a.values.size());
job.run_into(output, executor);
auto trace = job.trace();
auto ordinary = dmm::matmul(a, b, executor, "cannon", 16, 1);
```

## BSP operation plan

`plan.hpp` provides `BspPlan`, `Superstep`, `BufferId`, `MatrixView`,
`PayloadPart` and `BlockTriple`. Buffer IDs are typed and scoped to one
nonmovable plan. A matrix view carries its buffer, rows, columns, offset and
row stride; it borrows storage. Prefer the checked constructor and submatrix:

```cpp
BspPlan plan(P);
auto id = plan.allocate_buffer(node, rows * cols);
auto view = plan.matrix_view(id, rows, cols); // optional offset, stride
// stride=0 in matrix_view selects contiguous rows.
auto part = view.submatrix(row, col, height, width);
auto step = plan.add_superstep(Phase::Multiply);
step.add_zero(c);
step.add_gemm(a, b, c, {i, k, j});
```

`allocate_buffer(owner,words)` allocates actual storage immediately.
`buffer(BufferId)` returns a mutable span for host reset/gather; it must be
serialized with execution. Algorithm communication uses the operations below.

| Superstep operation | Meaning |
| --- | --- |
| `add_zero(target)` | Zero only the view's rectangle |
| `add_copy(source,destination)` | Copy equal, nonoverlapping rectangles on one node |
| `add_gemm(a,b,c,{i,k,j})` | Accumulate C += A×B using the common tile-32 kernel; infer dimensions and record actual FLOPs/coverage |
| `add_send(destination,payload)` | Pack/copy local payload for next-step delivery; infer source from ownership |
| `add_receive(source,target)` | Copy the previous step's message into local target(s); infer destination |
| `add_receive_sum(source,target)` | Add its components into local targets and count additions |

Phases are `InputConversion`, `Skew`, `Replication`, `Multiply`, `Reduction`,
`OutputConversion`. Only the newest step accepts operations. For each node,
append local/receive copies, then GEMMs, then sends; a backward stage is rejected.
Different nodes progress independently during plan construction. Execution uses
three blocking executor batches per logical superstep. S counts one step.

Messages match by **sending step, source node, destination node**. A pair can
send one logical message per step, consumed exactly once in the immediately next
step. There is no public tag, message ID or receive table. Self data stays local.
Use one ordered bundle when sending several components between the same pair:

```cpp
auto send = plan.add_superstep(Phase::Replication);
send.add_send(destination, {{a_tile, {a_row, a_col}},
                            {b_tile, {b_row, b_col}}});
auto receive = plan.add_superstep(Phase::Replication);
receive.add_receive(source, {local_a, local_b});
```

This is one message, with word count equal to both rectangles combined. Each
component is flattened row by row, then unpacked into its corresponding target;
strides/padding are not transmitted. The two `GlobalPosition` values annotate
trace coordinates only. The components and targets must have equal arity and
matching shapes in order. Swapping equal-shaped A/B targets cannot be inferred
by endpoint matching; their positional order is the author's responsibility.
A single view or `PayloadPart{view,origin}` also works. Dynamic lists use spans.

`finalize(original,padded,product_grid)` checks matching, component shapes,
regions and FLOP arithmetic, rejects unmatched sends/receives and prepares cached
message slots. A finalized plan is immutable. `PreparedMatmul` calls it for you.
`execute(executor)` interprets operation lists and records actual work/messages;
it does not generate machine code or sleep to emulate a network.

## Shared collectives and redistribution

`collectives.hpp` adds fixed binary trees to a plan:

```cpp
using TreeGroup = std::vector<std::size_t>; // ordered: root first
using NodePayloads = std::vector<std::vector<PayloadPart>>;
void add_tree_broadcast(BspPlan&, std::span<const TreeGroup>, const NodePayloads&);
void add_tree_reduce_sum(BspPlan&, std::span<const TreeGroup>, const NodePayloads&);
```

Lists are indexed by node and contain ordered components. A convenience overload
accepts one `PayloadPart` per node. Participants need matching shapes/origins and
correct ownership; nonparticipants may have empty lists. Groups must be disjoint;
rank r has parent `(r-1)/2`. All groups share levels. Depth d takes d+1 steps,
including the final receive. Singletons append no steps. Two operands can share
one broadcast message; reduction additions are counted automatically.

`redistribution.hpp` provides checked `grid_size(P,c)`, common
`padded_size(original,P)`, and:

```cpp
void add_rows_to_tiles(BspPlan&, const MatmulConfig&,
                      std::span<const MatrixView> rows,
                      std::span<const MatrixView> tiles,
                      std::span<const MatrixView> scratch = {});
void add_tiles_to_rows(BspPlan&, const MatmulConfig&,
                      std::span<const MatrixView> tiles,
                      std::span<const MatrixView> rows,
                      std::span<const MatrixView> scratch = {});
```

Rows cover all P nodes. With P=c*q², tiles occupy the first q² nodes, the base
layer; other entries may be empty. In-place self segments need separate h×b
scratch (h=N/P, b=N/q). Each direction appends send and receive steps with real
rectangular traffic. At P=1 reuse the same row/tile storage; no step is appended.

## Trace, memory and validation

`trace.hpp` defines records, `Counts`, `count`, `count_native`, `bsp_cost` and
phase timings. A `MessageRecord` has source, destination, words and ordered
`regions`. Executed products include the block triple and actual GEMM shape.
Counts sum per-step critical work/traffic; native excludes only conversion.
`validation.hpp` provides `validate_coverage(trace)` and FP64 `validate`.
Coverage checks are separate from numerical correctness.

Allocate node payloads through the plan. Move-only internal buffers and the
shared kernel live in `detail/memory.hpp` and `detail/gemm.hpp`; algorithms call
`add_gemm`. Retained message capacity, scratch and replicas count even when idle.
Trace peaks are per node; `Counts.peak_words` is their maximum. Host matrices,
oracles and metadata are outside this payload metric. See [model](../model.md).

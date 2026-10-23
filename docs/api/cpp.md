# C++ API

## Matrix and executor

`algorithm.hpp` provides owning `Matrix<T>{size,values}`, checked read-only
`MatrixBlockView<T>` and the compatibility `Algorithm` concept. Matrices must be
nonempty square storage; size arithmetic is checked. The experiment uses FP32
inputs and an independent FP64 reference.

`execution.hpp` provides synchronous `Executor::for_each(count,task)`,
`SerialExecutor`, `ThreadExecutor(workers)` and `make_executor(workers)` (zero
selects serial). Logical P is independent of the thread count. See
[executor lifecycle](../design.md#executors).

## Algorithm builder

`matmul.hpp` defines the entire algorithm extension contract:

```cpp
struct MatmulLayout {
    const std::size_t n, P, c; // n is the common padded size
    std::vector<std::size_t> a, b, result;
    CoverageGrid coverage;
};
using PlanBuilder = void (*)(BspPlan&, MatmulLayout&);

void build_cannon(BspPlan& plan, MatmulLayout& layout);
```

The builder allocates node-owned buffers and binds one ID per node in each of
`a`, `b` and `result`. Each needs capacity at least n²/P words. Inputs and result
must be distinct. Larger capacity permits row/tile reuse with shared conversion
scratch. The builder declares C zeroing and all communication/arithmetic phases,
then sets the coverage grid. It neither executes nor seals the plan. Missing,
foreign, undersized or aliased common row bindings are rejected by the workspace.

## Preallocated execution

```cpp
class PreparedMatmul {
public:
    PreparedMatmul(Matrix<float> a, Matrix<float> b,
                   std::size_t P, std::size_t c, PlanBuilder build);
    std::size_t size() const noexcept; // original size
    RunTimings run_into(std::span<float> output, Executor& executor);
    Trace trace() const;
    Traced run(Executor& executor); // allocate host output and return product/trace
};
```

The common workspace owns input copies, computes `padded_size(original,P)`,
invokes the builder and seals the result. `run_into` serializes host reset,
execution and cropped output gathering. Output needs exactly original_n² words.
Only the common-row-to-common-row execution is internally timed. First and
repeated execution allocate nothing; copying the trace is outside the timer.
`trace` throws before the first success or after a failed execution. A subsequent
successful run recovers the workspace. `run` keeps output execution and snapshot
under the same workspace lock.

```cpp
dmm::PreparedMatmul job(a, b, 16, 1, dmm::build_cannon);
std::vector<float> output(a.values.size());
job.run_into(output, executor);
auto trace = job.trace();
```

The C++ `Cannon{P,c}` compatibility facade offers `run` and `matmul` by creating
this workspace temporarily. There is no per-algorithm prepared class, generic
algorithm facade template, callback BSP runtime or standalone native-input mode.

## BSP operation plan

`bsp_plan.hpp` defines `BspPlan`. Buffer IDs refer to owned storage and
`BufferView{buffer,offset,stride}` describes rectangular local operands.

```cpp
BspPlan(std::size_t nodes);
std::size_t allocate(std::size_t node, std::size_t words);
std::span<float> buffer(std::size_t id);
bool owns(std::size_t node, std::size_t id) const;
std::size_t nodes() const;
std::size_t step(const std::string& phase);
void zero(std::size_t step, std::size_t node, std::size_t buffer);
void copy(std::size_t step, std::size_t node, BufferView from, BufferView to,
          std::size_t rows, std::size_t cols);
std::size_t send(std::size_t step, std::size_t source, std::size_t destination,
                 std::size_t tag, BufferView from, ElementRegion region);
void receive(std::size_t step, std::size_t node, std::size_t message,
             BufferView to, bool add = false);
void multiply_add(std::size_t step, std::size_t node, ProductRegion product,
                  BufferView a, BufferView b, BufferView c);
void seal(std::size_t original_size, std::size_t padded_size, CoverageGrid grid);
RunTimings execute(Executor& executor);
const Trace& trace() const;
```

Allowed phases are `input_conversion`, `skew`, `replication`, `multiply`,
`reduction`, `output_conversion`. Each step executes local/receive copies,
kernels and sends in three blocking batches. S still counts logical steps.
Messages need distinct source/destination and unique source/destination/tag per
step. Receive by returned message ID in the immediately next step, once only.
Views must fit node-owned storage; local in-place conversion uses explicit
scratch. `seal` checks undrained messages, global matrix rectangles and FLOP
arithmetic and preallocates cached message slots. A sealed plan cannot be changed.
`multiply_add` uses the common tile-32 kernel and records its actual FLOPs and
`ProductRegion{i,k,j,m,n,inner}`. `receive(add=true)` records reduction additions.
No algorithm manually reconstructs a trace or adds GEMM work a second time.

## Fixed tree collectives

`collectives.hpp` provides plan-building helpers:

```cpp
using TreeGroup = std::vector<std::size_t>;
void tree_broadcast(BspPlan&, std::span<const TreeGroup> groups,
                    std::span<const std::size_t> buffers, std::size_t tag,
                    std::span<const ElementRegion> regions);
void tree_reduce_sum(BspPlan&, std::span<const TreeGroup> groups,
                     std::span<const std::size_t> buffers, std::size_t tag,
                     std::span<const ElementRegion> regions);
```

Groups are disjoint ordered node lists. Position zero is root and position r
has parent (r−1)/2. Buffers cover all logical nodes and participants have equal
payload size; regions describe one rectangle per group. Tree levels share
supersteps across groups. Singleton trees append no operations. A four-node
three-word tree has three messages, nine total/critical words and three logical
steps per collective; reduce adds nine FLOPs.

## Common row and block conversion

`sharding.hpp` provides `padded_size(original,P)` and two plan-building helpers:

```cpp
void rows_to_blocks(BspPlan&, std::size_t n, std::size_t c,
                    std::span<const std::size_t> rows,
                    std::span<const std::size_t> blocks,
                    std::size_t tag, std::span<const std::size_t> scratch = {});
void blocks_to_rows(BspPlan&, std::size_t n, std::size_t c,
                    std::span<const std::size_t> blocks,
                    std::span<const std::size_t> rows,
                    std::size_t tag, std::span<const std::size_t> scratch = {});
```

Use n=ceil(original/P)*P. `grid_size(P,c)` requires P=c*q² and c divides q.
The base layer is the first q² IDs. Row buffers hold n²/P words; base tiles
hold n²/q². Aliased conversions need separate owned n²/(P*q)-word scratch per
node. P=1 uses the same row/block buffer and adds no conversion steps; otherwise
each direction has two steps. Only explicit messages cross nodes. No algorithm
implements host reset or gathering; `PreparedMatmul` handles those outside timing.

## Node payload ownership and peak memory

`memory.hpp` implements move-only internal `Buffer` storage and node memory
ledgers. All algorithm allocations go through `BspPlan::allocate`; only views of
owned storage may be used by operations. Allocated FP32 capacity, including
scratch and retained idle message slots, is counted. Message delivery transfers
cache accounting to the receiver until receive finishes. Failure recovery
restores the accounting before retry. Host matrices, metadata and whole-process
RSS are outside this payload metric. Memory aggregation takes maximum, not sum.

## Traces, counts and correctness

`bsp.hpp` exposes `Trace`, `SuperstepRecord`, `MessageRecord`, `ProductRecord`,
`ElementRegion`, `ProductRegion`, `CoverageGrid`, `Counts` and `Traced`.
`count` aggregates actual work/messages; `count_native` excludes only input/output
conversion. Critical values sum per-step maxima, with h=max(sent,received).
`bsp_cost(counts,{w,g,l})` evaluates w*W_crit+g*C_crit+l*S.

`coverage.hpp` checks the full block-triple partition exactly once. Ring uses
{P,P,1}; Cannon/2.5D use {q,q,q}. `validate` compares FP32 results with FP64 using
atol=1e-4, rtol=1e-3 and reports maximum absolute and Frobenius relative errors.
The trace is an audit record, not an independent numerical oracle.

## Timing and calibration

`Timing{total,compute,copy,synchronization}` and `RunTimings{end_to_end,phases}`
report the six fixed phases. Inactive phase times may be zero. The remaining
wall time after critical local compute/copy includes scheduling/instrumentation.
`calibration.hpp` supplies independent `microbenchmark` functions. The Python
measurement runner freezes coefficients before held-out full-matrix runs; see
[measurement protocol](../measurement.md).

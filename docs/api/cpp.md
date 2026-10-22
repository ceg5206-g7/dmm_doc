# C++ API

Headers live in `include/dmm/`. This page describes the native interface;
the [Python reference](index.md) is generated separately with pydoc.

## Matrices and block views

```cpp
template <typename T>
struct Matrix {
    explicit Matrix(std::size_t size);
    std::size_t size;
    std::vector<T> values;
    MatrixBlockView<T> block(std::size_t row, std::size_t col,
                            std::size_t block_size) const;
};

template <typename T>
struct MatrixBlockView {
    const T* data;
    std::size_t size;
    std::size_t stride;
    const T& operator()(std::size_t row, std::size_t col) const noexcept;
};
```

Matrices own zero-initialized row-major square storage. Construction rejects
zero dimensions and size arithmetic overflow using `sizeof(T)`. `block` rejects
malformed storage and empty/out-of-bounds blocks. Views are read-only, trivially
copyable borrowers; keep the source alive and its storage stable.

Cannon uses `Matrix<float>` at its host adapters and tracked node buffers
inside its common-shard and native schedules. Independent native test references use `Matrix<double>`.
Template definitions are in `algorithm.hpp`.

## Shared blocked GEMM

`include/dmm/gemm.hpp` provides `GEMM_TILE = 32` and:

```cpp
void gemm_add(std::size_t m, std::size_t n, std::size_t k,
              std::span<const float> a, std::size_t lda,
              std::span<const float> b, std::size_t ldb,
              std::span<float> c, std::size_t ldc);
```

This computes `C += A*B` for row-major FP32 rectangles with row strides in
words. It allocates nothing, supports tile tails, preserves input and padding
words, and visits k in ascending order per output element. Nonzero C is allowed;
C must not overlap A/B. Storage lengths and strides are checked before writing,
including integer overflow. All layouts must use this kernel and frozen tile.
Tests compare strided rectangles against an independent FP64 oracle.

## BSP runtime

`include/dmm/bsp.hpp` declares an algorithm-independent runtime of logical nodes
with private mailboxes. Communication is copied and counted, never delayed.

```cpp
class Node {
public:
    std::size_t id() const noexcept;
    std::size_t nodes() const noexcept;
    void send(std::size_t destination, std::size_t tag,
              std::span<const float> payload, std::optional<ElementRegion> region = std::nullopt);
    Buffer receive(std::size_t source, std::size_t tag,
                               std::size_t words);
    Buffer buffer(std::size_t words = 0) const;
    void compute(std::size_t flops) noexcept;
};

class Bsp {
public:
    explicit Bsp(std::size_t nodes);
    std::size_t nodes() const noexcept;
    Buffer buffer(std::size_t node, std::size_t words = 0) const;
    NodeBuffers buffers(std::size_t words = 0) const;
    std::size_t live_words(std::size_t node) const;
    void superstep(std::string label, Executor& executor,
                   const std::function<void(Node&)>& body);
    Trace finish();
};
```

`Bsp(nodes)` rejects zero nodes with `std::invalid_argument`. `superstep` runs
`body` once per node as one `Executor::for_each` batch, checks that every
message delivered by the previous superstep was received, records the
superstep and then delivers its messages. `finish` performs the same check for
the last superstep, returns the trace and starts an empty one.

A `Node` is the handle for one node during one superstep; only that node's task
uses it, and it cannot be constructed outside `Bsp`.

- `send` copies `payload` into a message delivered for the next superstep.
  A destination outside the machine throws `std::out_of_range`; sending to
  itself throws `std::invalid_argument`; a second message with the same
  destination and tag in one superstep throws `std::logic_error`.
- `receive` moves out the message from `(source, tag)` delivered at the end of
  the previous superstep. An invalid source throws `std::out_of_range`; a
  missing message or a length other than `words` throws `std::logic_error`.
- `compute` adds local FLOPs; a multiply-add counts as two.

An unreceived delivered message makes the next `superstep` or `finish` throw
`std::logic_error`.

## Traces, counts and cost

```cpp
struct MessageRecord {
    std::size_t source, destination, tag, words;
    std::optional<ElementRegion> region;
};

struct SuperstepRecord {
    std::string label;
    std::vector<std::size_t> flops;      // Indexed by logical node.
    std::vector<MessageRecord> messages;
    std::vector<std::size_t> peak_words; // Per-node peak for this step.
    std::vector<ProductRecord> products;
};

struct Trace {
    std::size_t nodes = 0;
    std::vector<SuperstepRecord> supersteps;
    std::vector<std::size_t> peak_words; // Per-node peak for this trace.
    std::size_t original_size = 0, padded_size = 0;
    CoverageGrid coverage;
};

struct Counts {
    std::size_t supersteps = 0;
    std::size_t total_flops = 0, critical_flops = 0;
    std::size_t total_words = 0, critical_words = 0;
    std::size_t messages = 0;
    std::size_t peak_words = 0;
    Counts& operator+=(const Counts& other) noexcept;
    friend bool operator==(const Counts&, const Counts&) = default;
};

Counts count(const SuperstepRecord& superstep);
Counts count(const Trace& trace);
Counts count_native(const Trace& trace);

struct CostParameters {
    double w = 1, g = 0, l = 0;
};

double bsp_cost(const Counts& counts, const CostParameters& parameters);
```

Payload sizes are FP32 words. Messages are recorded in the superstep that sent
them. For a superstep, `count` sums FLOPs, words and messages, and takes
`critical_flops` as the maximum node FLOPs and `critical_words` as the maximum
over nodes of $h = \max(\text{sent}, \text{received})$. Counting a trace adds
the per-superstep counts, so critical counts are sums of per-superstep maxima.

`bsp_cost` returns
`w * critical_flops + g * critical_words + l * supersteps`. Non-finite or
negative parameters throw `std::invalid_argument`. Python exposes these records
read-only through `dmm.bsp`, with `bsp_cost(counts, *, w=1, g=0, l=0)`.

## Algorithms

```cpp
struct Traced {
    Matrix<float> product;
    Trace trace;
};

template <typename T>
concept TracedAlgorithm = Algorithm<T> && requires(
    const T& algorithm, const Matrix<float>& a, const Matrix<float>& b,
    Executor& executor) {
    { algorithm.run(a, b, executor) } -> std::same_as<Traced>;
};

inline constexpr std::size_t GRID = 4;

struct Cannon {
    std::size_t P = 16, c = 1;
    Matrix<float> matmul(const Matrix<float>& a, const Matrix<float>& b,
                         Executor& executor) const;
    Traced run_native(const Matrix<float>& a, const Matrix<float>& b, Executor& executor) const;
    Traced run(const Matrix<float>& a, const Matrix<float>& b,
               Executor& executor) const;
};
static_assert(TracedAlgorithm<Cannon>);
```

`Algorithm<T>` is a C++20 concept requiring a const algorithm to accept two
FP32 matrices plus an executor reference and return exactly `Matrix<float>`
by value. `TracedAlgorithm<T>` also requires `run` to return the product and
its BSP trace as `Traced`. Algorithms use plain structs without a virtual base.

Cannon validates matching square storage, c=1 and square P=q*q. `run` uses
common row shards, zero padding to a multiple of P and traced conversions;
`run_native` requires dimensions divisible by q and executes only the q+1 body
[supersteps](../algorithms/cannon.md#supersteps-and-message-routes) on `Bsp(P)`; `matmul` returns `run(...).product`.

## Executors

```cpp
class Executor {
public:
    virtual ~Executor() = default;
    virtual void for_each(std::size_t count,
                         const std::function<void(std::size_t)>& task) = 0;
};

// SerialExecutor and ThreadExecutor implement Executor.
std::unique_ptr<Executor> make_executor(std::size_t workers);
```

`count` is the number of tasks; `task(index)` performs task `index` for indices
from zero through `count-1`. The call is synchronous and propagates task errors.
`Bsp` submits one batch per superstep with one task per node, so Cannon
submits eleven batches of 16 tasks by default on the common route. A zero task count does no work.

`make_executor(0)` creates serial execution. Positive counts create a thread
pool. `ThreadExecutor` starts workers at construction, provides `enter()` and
idempotent `close()`, and closes on destruction. Calls sharing a pool serialize
batches; closing waits for the active batch. See [lifecycle rules](../design.md#executors).

## Native correctness validation

```cpp
struct Error {
    double max_absolute;
    double frobenius_relative;
};

Error validate(const Matrix<float>& result, const Matrix<double>& reference,
               double atol = 1e-4, double rtol = 1e-3);
```

Validation checks dimensions and both storage lengths, finite tolerances and
values, and elementwise absolute/relative tolerance. Malformed arguments throw
`std::invalid_argument`; incorrect/non-finite values throw `std::runtime_error`.
The benchmark uses its Python NumPy validator; native tests use this function.

## Fixed tree collectives

`include/dmm/collectives.hpp` exposes:

```cpp
using TreeGroup = std::vector<std::size_t>;
void tree_broadcast(Bsp&, Executor&, std::span<const TreeGroup> groups,
                    NodeBuffers& values, std::size_t tag);
void tree_reduce_sum(Bsp&, Executor&, std::span<const TreeGroup> groups,
                     NodeBuffers& values, std::size_t tag);
```

Each group is an ordered, non-empty list of distinct node ids. The first is
root; rank r has parent (r-1)/2 and children 2r+1, 2r+2 where present. Groups
must be disjoint and ids in range. `values` has one entry per machine node;
a node task accesses only its own entry. Nonparticipants remain unchanged.
Broadcast overwrites nonroot entries with copied root data. Reduction requires
equal lengths within a group, sums children in rank order and leaves the result
at the root (other entries are not promised to remain original partial sums).
Validation completes before any superstep is executed.

Independent groups execute in the same supersteps. Broadcast visits depth 0
through the maximum depth; reduction visits the opposite order. Labels are
`replication` and `reduction`. The last step consumes the final messages and is
charged as a real superstep; singleton-only groups need zero supersteps.
Every edge uses `send/receive`; every reduction element addition records one
FLOP. For a four-node group with m words, each operation records 3 messages,
3m total words, 3m critical words and 3 supersteps. Reduction adds 3m total and
critical FLOPs. Counts describe this exact heap-tree schedule, not a substituted
idealized logarithmic formula. Callers choose tags; both collectives drain their
mailboxes before returning.

## Node payload ownership and peak memory

`memory.hpp` defines move-only `Buffer` and `NodeBuffers = std::vector<Buffer>`.
Create them with `Bsp::buffer(node, words)`, `Bsp::buffers(words)` (one per node),
or `Node::buffer(words)` inside a node task. Buffers expose size/data/begin/end,
indexing, `assign(count, value)`, `assign(span)` and `clear()`. They work as spans
with `send` and `gemm_add`. Copying a Buffer is disabled; send makes an explicit
counted copy. `receive` transfers a Buffer without copying. The ledger stays
alive if a buffer outlives the machine.

Counts track allocated FP32 payload slots, including vector capacity, with no
container/trace metadata. Same-size assign reuses storage; size-changing assign
counts the simultaneous old and new allocations before releasing the old one.
`clear` frees payload storage. Message delivery transfers accounting from the
sender's outbox to the receiver's inbox; receiving/moving into local storage
never registers a second allocation. Delivery order is the fixed source-id
order in BSP and independent of physical worker count.

`live_words(node)` reports current payload storage. Each superstep snapshots its
own per-node peak, and `finish` returns the lifetime per-node peaks, then starts
a new peak interval with live buffers as its baseline. `Counts.peak_words` is
the maximum node peak; combining counts takes max for memory, sums for work and
communication. Do not add individual buffer peaks or node peaks.

Cannon and tree collectives use these buffers. The current host-side Matrix
inputs/output and FP64 oracle are whole-program storage outside node payload
budgets. Future common row shards, conversions, packing and replicas must use
this same buffer API, so their simultaneously live payload is included. The
whole process resident memory and DRAM traffic are separate metrics.

## Common shards and conversion

`sharding.hpp` provides `RowShards{original_size, size, NodeBuffers values}` and
`BlockShards{original_size, size, c, NodeBuffers values}`. Construct row inputs
with `make_row_shards(machine, matrix)` and gather/crop with
`gather_row_shards(rows)`. `padded_size(n,P)` checks overflow and FP32 storage
size before allocation. Common n is padded to a multiple of P.

`rows_to_blocks(machine, executor, rows, c, tag)` converts to base-layer blocks;
`blocks_to_rows(machine, executor, blocks, tag)` converts the base C layer back.
For 2.5D the first q*q nodes hold base blocks and other layers start empty.
Both use two supersteps per conversion, or zero for P=1. Messages carry
`ElementRegion{row,col,rows,cols}`; its rectangle must match payload size and
must not overflow. Buffer dimensions and ownership are checked before execution,
including rejecting buffers from another machine. Round-trip tests cover P=1,
16,64,256, c=1 and the two valid c=4 proposal configurations.

`count_native(trace)` excludes exactly input_conversion/output_conversion labels
and takes memory peaks from those retained steps. `count(trace)` includes all
stages. Each node records packing and local output storage in the same memory
ledger as computation. Host scatter/gather adapters do not masquerade as logical
communication; the actual row/block redistribution is always explicit messages.

## Executed multiplication coverage

`Node::multiply_add(product, A, lda, B, ldb, C, ldc)` calls the shared blocked
GEMM, then records its successful execution and FLOPs together. The product is
`ProductRegion{i,k,j,m,n,inner}`: uniform block indices and actual GEMM dimensions.
Failed bounds checks do not create records or charge work. Each superstep's
`products` contains `ProductRecord{node,product}`; these metadata are excluded
from payload memory accounting.

`Trace.coverage = CoverageGrid{row_parts,inner_parts,col_parts}` defines the
uniform partition of padded_size. Cannon uses q,q,q; a row ring uses P,P,1;
2.5D uses q,q,q across all layers. The shared `validate_coverage(trace)` in
`coverage.hpp` independently checks node ids, indices, actual GEMM shapes,
per-node recorded FLOPs and exactly one execution of every (i,k,j) triple.
It never expands n³ scalar terms; storage scales with the number of block
triples. Invalid partition metadata throws invalid_argument; missing, repeated
or inconsistent executed blocks throw runtime_error.

Python exposes readonly records and `dmm.bsp.validate_coverage(trace)`. Use it
alongside the independent FP64 oracle and message checks. Tests deliberately
remove, duplicate and alter executed records to verify the validator fails.
The model pipeline must invoke this validation before persisting trusted counts.

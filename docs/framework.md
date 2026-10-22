# Framework implementation contract

The eight accepted implementation decisions below have code, tests and committed
documentation. The [versioned proposal](proposal.pdf) includes the approved
C++20 amendment. Full three-layout experimental acceptance remains a separate
integration gate; implemented infrastructure does not establish that comparison.

- [x] Shared rectangular FP32 blocked GEMM: tile 32, row strides, tails, no allocation.
- [x] Fixed binary-tree broadcast and FP32 sum reduction through copied BSP messages.
- [x] Common per-node row shards and traced conversion to/from 2D/base-layer layouts.
- [x] Live per-node payload memory including owned input/output, packing and messages.
- [x] Actual block/interval multiplication coverage with an independent validator.
- [x] Common zero padding to a multiple of P, cropping and original/padded metrics.
- [x] Trace persistence, model/profile scans, two budgets, layout charts, prepared
      timing, 5/15 median/IQR, independent calibration and held-out residual reports.
- [x] C++20 amendment of the proposal (the implementation already uses C++20).

Each YAML case requires layout/P/c. Ring uses c=1; Cannon uses c=1 and P=q²;
2.5D uses P=c*q² and c dividing q. P can exceed worker count H: the executor
batches logical nodes. Dataset/oracle I/O and host-side adapters are outside
logical-node budgets. Node-owned allocated capacity remains counted when idle;
whole-process RSS is a different metric.

The six phases are input_conversion, skew, replication, multiply, reduction,
output_conversion. Native body costs exclude only the first and last. Packing
is timed and counted in payload memory; only node-to-node traffic is charged as
communication. No sleep emulates network time. Executed GEMMs record dimensions
and block triples through the shared kernel hook, checked independently without
enumerating scalar n³ terms.

[Prepared execution](api/cpp.md#preallocated-execution) shares operation scheduling,
message caches, tree collectives and common layout conversions. Cannon is its
reference adapter. Tests verify no allocation on caller/worker threads, including
the first execution, and recovery after an interrupted executor batch. Prepared
and allocation-based routes use their own actual peaks; measurements and budgets
must use the same route's trace.

[Model analysis](model.md) replays checked saved traces without loading matrices.
[Prepared measurement](measurement.md) freezes independent local w/g/l before
held-out full-matrix runs and preserves every raw sample/residual. Source/build,
settings, input hashes, physical/logical CPUs, binding policy and load are saved.
See the [verification record](verification.md) for actual runs and limitations.

## Remaining integration gate

- [ ] Implement ring and 2.5D (**assigned to teammates**).
- [ ] Execute the full `configs/model.yaml`: 24 actual configurations, 192 model
      scenarios and 144 budget-selection cells with all required candidates.

The frozen profile/cardinality test specifies that target; it does not replace
execution. Missing algorithms fail preflight. No unavailable algorithm is
represented by a fabricated or analytic execution. Shared framework verification
has passed for real Cannon cases and independent c=4 conversion tests; the full
three-layout comparison awaits teammate implementations.

Each implementation milestone was tested and committed in both repositories.
Teammates follow the [algorithm/prepare contracts](contribute.md), using the shared
kernel, communication, collectives, conversions, coverage and measurement APIs.

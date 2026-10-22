# Framework implementation contract

The accepted scope is the following eight decisions. A checked item has code,
tests and documentation; unchecked items remain required work. The [versioned proposal PDF](proposal.pdf) includes the approved C++20 amendment.

- [x] Shared rectangular FP32 blocked GEMM: tile 32, row strides, tails, no allocation.
- [x] Fixed binary-tree broadcast and FP32 sum reduction through BSP messages.
- [x] Common per-node row shards and traced conversion to/from 2D/base-layer layouts.
- [x] Live per-node payload memory including owned input/output and message buffers.
- [x] Actual block/interval multiplication coverage with an independent validator.
- [x] Common zero padding to a multiple of P, cropping and original/padded metrics.
- [ ] Trace persistence, 24 executed configurations, 192 model scenarios, two budgets,
      layout charts, preallocated measurement, 5/15 median/IQR, calibration and residuals.
- [x] C++20 amendment of the proposal (the implementation already uses C++20).

Existing P/c case configuration is shared across YAML, Python and C++. Each YAML
case requires layout/P/c. `ring` uses c=1; Cannon uses c=1 and P=q²; 2.5D uses
P=c*q² and c dividing q. Ring and 2.5D still require implementation.

The target end-to-end phases are input_conversion, skew, replication, multiply,
reduction, output_conversion. Native body costs exclude only the first and last.
Local packing is timed and included in payload memory; only node-to-node traffic
is charged as communication. No sleeps emulate network time.

Each milestone is tested and committed in the documentation submodule and main
repository. Future model data must be backed by real traces; unavailable
algorithms must never be replaced by fabricated or purely analytic execution.

Memory accounting now covers Cannon local tiles and all BSP/tree message buffers.
Common row-shard/conversion/packing buffers now use these same tracked buffers.
Host-side matrices and the oracle remain outside logical-node budgets.

Executed products now record actual GEMM dimensions and uniform block triples
at Node::multiply_add. The shared validator supports q/q/q and P/P/1 grids
without enumerating scalar n³ terms. Algorithm/model integration must call it.

The statistics subtask now defaults to 5 warmups/15 repeats, reports median/IQR
with linear quartiles, and uses matching median baselines. The experiment
milestone remains unchecked until persistence, model sweep, preallocated timing,
calibration and residual validation are finished.

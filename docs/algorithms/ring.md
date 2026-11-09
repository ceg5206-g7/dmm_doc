# 1D Ring

Use `layout="ring"` with any positive logical node count `P` and `c=1`.
The Python module `dmm.algorithms.ring` provides `validate_size`, `matmul`,
`matmul_traced` and `prepare`. The C++ function `build_ring` in
`src/algorithms/ring.cpp` declares the schedule on the shared `BspPlan`.
The ordinary and prepared entry points use the same schedule and FP32 GEMM.
[Shared algorithm interface](overview.md#algorithm-interface)

## Ownership and padding

The common workspace pads the original size `n` to `N = ceil(n/P) × P` and
sets `h = N/P`. Node `r` owns an `h × N` row shard of A, its output shard of C
and a buffer initially holding row shard `r` of B. A and C stay on their
owners. B moves through explicit copied messages, using the same node-local
buffer for each incoming shard. The common row shards are already Ring's
working layout, so native and common-route counts coincide. The workspace
gathers and crops C to the original size.

## Superstep schedule

The builder emits one `multiply` superstep for each round `t=0,…,P−1`.
Node `r` computes with B shard `k = (r−t) mod P`, using the non-negative
remainder. In round zero the node zeros C and uses its local B shard. In
later rounds it first receives from `(r−1) mod P`. It then accumulates
`A_r[:, k*h:(k+1)*h] × B_k` into its C shard with the shared GEMM.
Every round except the final one sends the current B shard to `(r+1) mod P`.
At `P=1`, the plan performs one GEMM and sends zero messages.

The message trace records the B rectangle with origin `(k*h, 0)` and shape
`h × N`. Each executed GEMM records `(r,k,0)` in a `P × P × 1` coverage
grid. As `t` runs over the ring, every node sees every `k` exactly once.

## Counts and validation

| Metric | Native and common routes |
| --- | ---: |
| Supersteps | `P` |
| Total FLOPs | `2N³` |
| Critical FLOPs | `2N³/P` |
| Messages | `P(P−1)` |
| Words per message | `N²/P` |
| Total words | `(P−1)N²` |
| Critical words | `(P−1)N²/P` |

The counts use padded `N`. The hand-computed `N=6, P=3` case has two
communication rounds with three 12-word messages each, followed by a
compute-only round. The resulting total is 72 words and the critical count
is 24 words. The native tests also use distinct B rows to verify payload
movement through both shifts and compare outputs with an FP64 oracle.

Three resident row shards use `3N²/P` words per node. The shared runtime
also counts message-buffer capacity in its payload ledger. The formal main
grid recorded a peak of `5N²/P` words per node. See the layout comparison
for the effects on communication and budgets. [Results and Discussion](../results.md)

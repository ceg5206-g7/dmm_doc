# Cannon's Algorithm

The Python entry points are `matmul` and `matmul_traced` in
`python/dmm/algorithms/cannon.py`; C++ arithmetic is in `src/cannon.cpp`.
All arithmetic uses the shared blocked `gemm_add` kernel (tile 32).

## Cannon's logical block grid

P=q², c=1. Node (i,j) has id qi+j. Each holds one A/B/C square block of
side b=n_padded/q. In round s, its inner-product block is k=(i+j+s) mod q.
Each k is visited once. The physical worker count H is independent of P.

## Common input/output and padding

The default route first creates common contiguous row shards. It pads zeros to
n_padded=ceil(n/P)*P, assigns n_padded/P rows per node, and records original_size
and padded_size in the trace. Both inputs are converted from those node-private
row shards to the native 2D blocks through real BSP messages. Conversion never
reads another node's array. Packing temporaries, outgoing copies, incoming
buffers and output allocation use tracked node payload storage.

After multiplication, C blocks convert back to the same row distribution; the
host output adapter gathers and crops only after the common row-shard result
exists. Host array copies are outside logical-node communication counts, while
the current convenience matmul timing includes those copies and allocation.
The preallocated measurement path is a remaining milestone.

## Supersteps and message routes

For P>1 the default call executes:

1. Two `input_conversion` supersteps for A (send/pack, then receive/unpack).
2. The same two supersteps for B; consumed input shards are released.
3. `skew`: A_ij sends to (i,j-i), B_ij to (i-j,j), modulo q. Self-owned data stays local.
4. q `multiply` steps: receive prior messages, accumulate A_ik B_kj, record 2b³
   FLOPs, then shift A left and B up except on the final round.
5. Two `output_conversion` steps convert C to common row shards. A/B tiles
   have been released before output conversion.

Every received message is consumed in the following superstep. Conversion
messages carry global row/column rectangles; skew and shift messages carry
operand block rectangles. P=1 conversion needs no communication/supersteps;
the existing skew/multiply body has two steps.

## Hand-computed communication counts

For P=16 and n=16 (no padding), the body has five steps:

| Superstep | Messages | Total words | max h | max F |
| --- | --- | --- | --- | --- |
| skew | 24 | 24b² | 2b² | 0 |
| multiply 0–2 | 32 each | 32b² each | 2b² | 2b³ |
| multiply 3 | 0 | 0 | 0 | 2b³ |
| Body total | 120 | 120b² | 8b² | 8b³ |

Here b=4. Each conversion retains one four-word segment on its owner and sends
12 words per node, in 48 messages total. Thus the common route records S=11,
F_total=8192, W_crit=512, C_total=2496, C_crit=164, messages=264. The body records
S=5, C_total=1920 and C_crit=128. Tests independently verify these values.

## Native and end-to-end metrics

Use `count(trace)` for the complete common route and `count_native(trace)` for
the same execution with only input/output conversion excluded. Skew, replication
and reduction (where relevant to other layouts) remain part of native-body cost.
This comparison uses identical actual dimensions and arithmetic.

`matmul_traced(..., native_layout=True)` and C++ `Cannon::run_native` provide an
explicit native-input adapter for schedule tests. They do not pad and require
n divisible by q. They load native blocks before the body and assemble a host
output afterwards; they are not the common-input end-to-end metric. When comparing
with a padded common execution, supply the same padded input size.

## Input and output contract

- Matching finite, non-empty square FP32 NumPy inputs.
- Positive square P, c=1; any positive original n on the common route.
- Read-only and strided inputs are accepted and preserved.
- A fresh, cropped FP32 output; traced calls return actual executed metadata.
- FP32 stage ordering depends on the padded layout. Validation compares against
  an independent FP64 oracle with the agreed tolerance, not universal bitwise equality.

Node storage uses move-only tracked buffers and explicit messages. The algorithm
object carries P/c without mutable execution state. Executed block triples and actual GEMM shapes are recorded by Node::multiply_add
and can be checked with validate_coverage. The 1D/2.5D algorithms remain pending. See [framework milestones](../framework.md).

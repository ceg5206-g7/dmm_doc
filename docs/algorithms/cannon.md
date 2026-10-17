# Cannon's Algorithm

The Python entry point is `python/dmm/algorithms/cannon.py`; computation lives in
`src/cannon.cpp`. See the [overview](overview.md) for availability.

## Cannon's logical block grid

Each $N \times N$ input is divided into a fixed $4 \times 4$ grid with block
dimension $b=N/4$. Output block $(i,j)$ accumulates four products:

$$
k_s=(i+j+s)\bmod 4,\qquad
C_{ij}=\sum_{s=0}^{3}A_{i,k_s}B_{k_s,j}.
$$

This represents initial row/column alignment followed by four logical stages.
For output block (1, 2), the shared indices are 3, 0, 1 and 2. Each index is
visited exactly once. Inputs are not physically shifted.

## Code path

1. The binding validates NumPy inputs and copies A/B into `Matrix<float>`.
2. `build_tasks` prepares 16 tasks in output-block row-major order. Each task
   holds its output coordinates and four pairs of `MatrixBlockView<float>`.
3. One synchronous `Executor::for_each` batch computes all tiles. Each task
   allocates a zeroed FP32 tile and visits its four stages, using scalar
   row/shared-index/column loops within each block product.
4. Cannon assembles the tiles into a new `Matrix<float>`. The binding copies
   this product into an independent FP32 NumPy array.

All four stages for one output block run inside one task. There is no separate
round-execution API or barrier between stages. Work is
$16\cdot4\cdot(N/4)^3=N^3$ scalar products. Tile assembly and matrix storage
scale quadratically.

## Views and workers

`Matrix<T>::block(row, col, size)` creates a read-only view of a square block.
It borrows the matrix's storage with its original row stride. Construction checks
storage shape and bounds; debug builds assert element indices. It performs no
allocation or element copying. The source must stay alive without reallocating
its storage while a view is used.

The logical grid is independent of thread count. Workers can compute several
tiles and have no permanent grid coordinate. Views share local process memory;
there is no serialization, distributed topology or network communication.
Communication simulation will require an explicit extension of this model.

## Input and output contract

- Matching, finite, non-empty square NumPy arrays of dtype `float32`.
- Positive integer dimension divisible by four; unsupported sizes fail preflight.
- Arbitrary strides and read-only arrays are accepted and preserved.
- A fresh FP32 result with independent storage is returned on every call.
- FP32 rounding depends on the prescribed stage order. Correctness uses an
  independent FP64 reference with [numerical tolerances](../experiments.md#correctness-and-arithmetic).

The algorithm object is stateless. It borrows A/B and an executor for each call;
it owns neither persistent matrices nor workers. Repeated calls do not alter
previously returned products.

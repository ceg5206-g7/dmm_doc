# 2.5D Replicated Cannon

Use `layout="2.5d"` with positive integers `P=cq²` and `c` dividing `q`.
The Python module `dmm.algorithms.two_point_five_d` provides `validate_size`,
`matmul`, `matmul_traced` and `prepare`. The C++ function
`build_two_point_five_d` in `src/algorithms/two_point_five_d.cpp` declares
one schedule on the shared `BspPlan`. [Shared algorithm interface](overview.md#algorithm-interface)

## Layout and ownership

The common workspace pads `n` to `N = ceil(n/P) × P`. All `P` nodes own
common row shards of height `N/P`. The working layout has `c` layers of
`q × q` nodes, with ID `z*q*q + i*q + j` for node `(z,i,j)`. Each layer
stores square A, B and C tiles of side `b=N/q`. Layer zero is the base
layer and supplies the roots for replication and reduction.

For `c>1`, tile storage is separate from the common row shards. At `c=1`,
the tiles fit the row-shard buffers and the conversion uses scratch for
overlapping local views, as in Cannon.

## Superstep schedule

1. **Input conversion.** Redistribute the two common row-sharded inputs to
   A and B tiles on the base layer. For `P>1` this takes two supersteps per
   input.
2. **Replication.** For each tile coordinate `(i,j)`, broadcast A and B
   together from the base layer along a fixed binary tree of layers. The
   tree groups are disjoint and use the same supersteps.
3. **Skew.** Let `R=q/c` and let layer `z` have offset `o=zR`. Zero the
   partial C tiles. Within layer `z`, move A left by `i+o` and B up by
   `j+o`, modulo `q`. A tile with zero displacement stays local.
4. **Multiplication.** Each layer performs `R` shared rounds. In round
   `t`, node `(z,i,j)` accumulates the product indexed by
   `k=(i+j+o+t) mod q`. Between rounds, A shifts left and B shifts up by
   one tile. The plan records a product triple `(i,k,j)` for each GEMM.
5. **Reduction.** Sum the partial C tiles from the leaves of the same
   binary tree to the base layer. Each received tile contributes `b²`
   FP32 additions to its parent's work count.
6. **Output conversion.** Convert the base-layer C tiles to the common
   row layout. The workspace gathers and crops the result.

For fixed `(i,j)`, the offsets `zR+t` cover `0,…,q−1` exactly once over
the layers and rounds. This gives one contribution for every `(i,k,j)`
in the `q × q × q` coverage grid. The layers share the same multiply
supersteps, so there are `R` such supersteps. At `c=1`, the tree groups
are singletons, and the common conversions, skew and multiply schedule
match Cannon.

## Executed phase counts

The two proposal grids at `c=4` have `R=1` at `P=64` and `R=2` at
`P=256`. The table gives critical words in units of `N²/P`. All counts
come from the executed main-grid traces.

| Phase | P=64 supersteps | P=64 critical words | P=256 supersteps | P=256 critical words |
| --- | ---: | ---: | ---: | ---: |
| Input conversion | 4 | 8 | 4 | 8 |
| Replication | 3 | 24 | 3 | 24 |
| Skew | 1 | 8 | 1 | 8 |
| Multiply | 1 | 0 | 2 | 8 |
| Reduction | 3 | 12 | 3 | 12 |
| Output conversion | 2 | 4 | 2 | 4 |
| Common route | 14 | 56 | 15 | 64 |

The shared trace counts `2N³` GEMM FLOPs and `(c−1)N²` reduction
additions. Native-body counts exclude only input and output conversion.
The retained workspace includes the row shards, working tiles and cached
messages. Both formal `c=4` grids recorded a logical-node peak of
`51N²/P` words.

Tests check numerical agreement with the FP64 reference, exact product
coverage, the hand-computed phases at `P=8, c=2`, and degeneration to
Cannon at `c=1`. The formal results compare the payload peaks and critical
communication with the other layouts. [Results and Discussion](../results.md)

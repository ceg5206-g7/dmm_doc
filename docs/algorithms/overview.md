# Algorithm Overview

## Implementation status

| Configuration name | Status | Size requirement |
| --- | --- | --- |
| [`cannon`](cannon.md) | Implemented | Square P=q*q, c=1; any positive n with common padding |
| [`ring`](ring.md) | Implemented | Positive P, c=1 |
| [`2.5d`](two_point_five_d.md) | Implemented | P=c*q*q and c divides q |

All three layouts use common zero padding for any positive original size.
The [formal results](../results.md) compare their executed traces and budgets.

## Algorithm interface

Each implementation exposes `validate_size(size, *, P, c)`, `matmul`,
`matmul_traced` and `prepare`.
The runner supplies data, execution settings, timing, and correctness checks.
See [System design](../design.md#dataset-and-algorithm-boundaries) for the shared
contract and [Adding an algorithm](../contribute.md) for the implementation and
testing workflow.

## Adding documentation

Place each algorithm's detailed documentation in `algorithms/<name>.md`, as in
[Cannon](cannon.md). Add its page under **Algorithms** in `doc/mkdocs.yml` and
update the status table above.

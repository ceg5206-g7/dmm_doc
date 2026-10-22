# Algorithm Overview

## Implementation status

| Configuration name | Status | Size requirement |
| --- | --- | --- |
| [`cannon`](cannon.md) | Implemented | Square P=q*q, c=1; any positive n with common padding |
| `ring` | Not implemented | Positive P, c=1 |
| `2.5d` | Not implemented | P=c*q*q and c divides q |

Selecting an unimplemented algorithm fails before an experiment starts. Each
implemented algorithm has its own page describing its size rules, computation,
validation, and performance limits.

## Algorithm interface

Each implementation exposes `validate_size(size, *, P, c)` and `matmul(A, B, executor, *, P, c)`.
The runner supplies data, execution settings, timing, and correctness checks.
See [System design](../design.md#dataset-and-algorithm-boundaries) for the shared
contract and [Adding an algorithm](../contribute.md) for the implementation and
testing workflow.

## Adding documentation

Place each algorithm's detailed documentation in `algorithms/<name>.md`, as in
[Cannon](cannon.md). Add its page under **Algorithms** in `doc/mkdocs.yml` and
update the status table above. Pages for unimplemented algorithms can be added
when their implementations are available.

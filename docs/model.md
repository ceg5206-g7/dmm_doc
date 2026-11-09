# Trace capture and model analysis

## Capture real data once

```sh
uv run dmm generate --config configs/dataset_smoke.yaml
uv run dmm model --config configs/model_smoke.yaml
```

Use fresh data/output paths if a previous run already exists. Dataset files and
model reports are never overwritten. The smoke profile executes four real
Cannon configurations (n=4/12, P=4/16), then computes eight cost scenarios and
sixteen budget-selection cells. This small grid intentionally includes padding
and infeasible memory budgets; it is a pipeline check, not a performance result.

Each configuration executes real FP32 inputs through `matmul_traced`, validates
against the saved independent FP64 oracle, and checks executed block coverage.
Capture verifies its independently reconstructed counts against both native
`count(trace)` and `count_native(trace)` before saving the trace. No wall-clock
capture duration is placed in a measurement column.

The trace directory contains:

- One JSON per n/layout/P/c with original/padded dimensions, H, per-node FLOPs,
  message endpoints/word counts/ordered component rectangles, actual product records,
  per-step/node peaks, correctness metrics and input content hashes.
- `correctness.csv` with one numerical summary per real configuration.
- A completion manifest with the exact ordered case/size selection, dataset
  manifest, compiler/build type, Git revision/dirty state and file checksums.

New captures use trace schema **2**: each logical message has `source`,
`destination`, `words`, and a nonempty `regions` list in payload order. There is
one message per ordered node pair per sending step, and words equals the sum of
component areas. No tag participates in matching. Schema **1** historical traces
with `tag` and a single `region` remain readable; their original counts, peaks
and provenance are preserved. The completion manifest stays schema 1, separately
versioned from its trace files. `python/dmm/trace.py` owns serialization,
independent validation and hash-checked loading; model and measurement share it.
A real pre-migration schema-1 Cannon fixture is tested through offline replay.

A failed capture may leave partial files. Without a complete manifest and every
expected trace, model analysis refuses to proceed. Missing algorithms fail
preflight before a new report directory is created. Unimplemented layouts are
never replaced by analytic or fabricated traces.

## Sweep saved traces

Create another YAML with a new output_dir and changed gammas/lambdas/budgets,
keeping cases, sizes, group and workers the same, then run:

```sh
uv run dmm model --config configs/my_model.yaml --traces results/model_smoke/traces
```

This path does not load matrices, import algorithm implementations or rerun
multiplications. It checks completeness, file hashes, message uniqueness and
bounds, matrix intervals, actual block coverage, work counts and memory lower
bounds before recomputing critical-phase counts from the saved steps. Execution
settings and H come from the captured run, not a hypothetical new machine.

## Cost and budget definitions

With w=w0, `T_model` stores normalized `T_BSP/w0` in FLOP-equivalents:

```text
T_model = W_crit + gamma*C_crit + lambda*S
M_cap = mu*n_original²/P words
```

Counts use actual padded work. F_total and C_total are global totals, while
W_crit/C_crit sum per-superstep node maxima. `peak_words` takes the largest
simultaneously live payload peak of any node, never the sum of node peaks.
The model assumes equal-speed nodes and an ideal fully connected, symmetric
full-duplex network without congestion or overlap. These are predictions under
those assumptions; memcpy bandwidth is not network bandwidth.

Budget levels filter the same costs and do not add matrix executions. Native
body costs exclude only common input/output conversion; native memory peaks
are tested against the same budget separately. Ties are retained. Cells with
no feasible candidate explicitly say infeasible.

## Outputs and plots

- `model.csv`: one end-to-end row per candidate/gamma/lambda. Columns include
  n, n_padded, P, H, layout, c, phase, F_total, W_crit, C_total, C_crit, S,
  peak_words, messages, gamma, lambda, T_model, t_measured, gflops, speedup,
  parallel_efficiency. The measured time and three performance fields stay empty.
- `phase_model.csv`: per-phase and native-body costs from the same execution.
- `layout_selection.csv`: budget cap, feasible/infeasible candidates and all
  minimum-cost native/end-to-end choices.
- `model.json`: formula, units, assumptions, cardinalities, analysis Git revision
  and source-manifest hash.
- `layout_selection.png`: faceted by n, lambda and budget, gamma horizontally
  and logical P vertically; available candidates, infeasible cells and ties are explicit.
- `cost_breakdown.png`, `conversion_breakdown.png`, `native_vs_common.png`:
  one labelled representative slice at the largest n/gamma/lambda. All other
  scenarios remain in the CSVs.
- `peak_memory.png`: payload peak normalized by n²/P with each budget marked;
  raw word counts are annotated. This is not whole-process RSS or DRAM traffic.

## Full proposal profile

`configs/model.yaml` freezes 3 sizes times 8 layout/P/c cases = 24 actual
configurations. Four gammas and two lambdas produce 192 model rows; two budgets
produce 144 layout-selection cells. Cardinality tests verify this plan.

The formal run at `9fc781b` captured all 24 configurations, including
`P=256,c=4`. The [results](results.md) compare native and common-route counts
and apply both budgets to those traces. The prepared measurement command
provides separate calibration and held-out residual checks using the same
trace format. [Measurement protocol](measurement.md)

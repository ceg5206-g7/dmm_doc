# Framework verification record

## Complete Cannon pipeline (2026-10-23)

Source: `bd85893db2a75a67b4504cd952e30e3546632097`, captured with a clean parent
and documentation working tree. This run checks the current endpoint-matched
plan runtime and formal measurement pipeline together. The host reported eight
physical cores; the fresh Release build used Apple LLVM 21.0.0
(`clang-2100.3.34.2`).

All 18 stages returned their expected exit codes: package installation, fresh
Release configure/build, CTest, Python/documentation tests, strict MkDocs build,
generation, benchmark, analysis, two model captures, two formal measurements,
two offline replays and three rejection checks.

- **77 C++ tests and 507 Python/documentation tests passed**; strict MkDocs passed.
- Dataset: seed 5206, group 0, FP32 range [0,255], original sizes
  17/65/256/257/512/1024, with independent NumPy FP64 references.
- FP64 validation passed for all model captures and measured repetitions at
  `atol=1e-4`, `rtol=1e-3`; traces retain maximum absolute and Frobenius relative
  errors and exact block coverage.

### Executed configurations and reports

| Stage | Configuration | Result |
| --- | --- | --- |
| Convenience benchmark and analysis | n=256/512/1024, P=1, c=1; serial and H=4 | 90 repeat rows, 6 median/IQR summaries |
| Main model capture | n=256/512/1024, P=16/64/256, c=1, H=1 | 9 executions, 72 modeled scenarios, 144 budget-selection cells |
| Padding model capture | n=17/65/257, P=16/64/256, c=1, H=1 | 9 executions, 9 modeled scenarios, 18 budget-selection cells |
| Formal measurement | n=256/512/1024; P=1/H=1 and P=4/H=4 | 5 warmups and 15 repeats per configuration; 90 measured products, 630 phase/end-to-end sample rows |
| Main-trace offline replay | gamma=3, lambda=700; saved P=16/64/256 traces | 9 modeled scenarios, 18 budget-selection cells |
| Measurement-trace offline replay | gamma=10, lambda=100; saved P=4/H=4 traces | 3 modeled scenarios, 6 budget-selection cells |
| Early rejection | measure P=16/H=4; model ring; model 2.5D | Each rejected before creating its output directory |

The main scan uses gamma=1/10/100/1000, lambda=100/10000 and memory budget
multipliers mu=8/32 (M_cap=mu*n_original²/P words); the padding scan uses
gamma=1, lambda=100 and the same budgets. Model
capture deliberately uses P>H for correctness and counts. Formal measurement
uses P<=H. The convenience benchmark includes allocation and host wrappers;
it is separate from the formal prepared measurements and their serial baseline.

The 18 captured main/padding traces were checked against independent Cannon
formulas. With padded N=ceil(n/P)*P, q=sqrt(P), b=N/q and h=N/P, both native and
common routes have F_total=2N³ and W_crit=2N³/P. The native route has S=q+1,
2(q−1)(q+P) messages, C_total=2(q−1)(q+P)b² and C_crit=2qb². Common input/output
conversion adds six steps, 3P(q−1) messages, 3P(q−1)hb total words and
3(q−1)hb critical words. The checks also validate q³ block products, saved/native
count agreement, padding and budget decisions.

Both offline replays succeeded with deliberately nonexistent matrix paths.
All model-only rows, including replay rows, leave measured time and performance
fields blank. Formal GFLOP/s, speedup and efficiency were independently
recomputed from the saved medians, original n and actual H.

### Formal end-to-end measurements

Times are seconds. Signed residual = (measured−predicted)/measured. The threaded
run explicitly used the completed same-input P=1/H=1 report via `--baseline`.
Each context froze its own independent microbenchmark calibration before
matrix measurements; no matrix-specific fit was performed.

| P/H | n | Median | IQR | GFLOP/s | Speedup | Efficiency | Relative residual |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1/1 | 256 | 0.002370750 | 0.000159417 | 14.15 | 1.000 | 100.00% | −0.11% |
| 1/1 | 512 | 0.019052667 | 0.000041167 | 14.09 | 1.000 | 100.00% | 0.35% |
| 1/1 | 1024 | 0.168290209 | 0.000507000 | 12.76 | 1.000 | 100.00% | 9.74% |
| 4/4 | 256 | 0.001184209 | 0.000162230 | 28.33 | 2.002 | 50.05% | 13.77% |
| 4/4 | 512 | 0.005716000 | 0.000152082 | 46.96 | 3.333 | 83.33% | 2.86% |
| 4/4 | 1024 | 0.041657750 | 0.000815145 | 51.55 | 4.040 | 101.00% | 0.16% |

Efficiency is speedup/H and is not clamped: the 1024 result remains 101.00%.
Cache effects and measurement variation could contribute; this run does not
isolate their causes or establish a fixed speedup.

### Evidence and scope

Raw artifacts are retained locally at
`results/cannon_pipeline_20261023__h524qgo/` (Git-ignored, not published with this
site). `validation.json` records source/build, stage commands and exit codes,
trace summaries, metrics and SHA-256 checksums for 119 artifacts; all 119 were
rechecked before documenting the run. The directory contains exact YAMLs in
`configs/`, command logs in `logs/`, benchmark CSV/analysis, model and measurement
reports, traces and 24 figures. The layout-selection and threaded residual
figures were visually checked.

To repeat the method, use the saved YAMLs with fresh data/output paths and run
`generate`, `bench`, `analyze`, `model`, then serial `measure` and threaded
`measure --baseline <serial-report>`. Replay with `model --traces <trace-dir>`;
the [model](model.md) and [measurement](measurement.md) guides give the CLI and
output contracts. Existing reports are immutable and cannot be overwritten.

This completes Cannon's pipeline verification. Selection plots explicitly name
Cannon as the only candidate. Ring, 2.5D and the full three-layout
24-execution/192-scenario comparison remain pending teammate integration; these
results do not establish a winner among three layouts. Earlier records below
retain their original revisions, datasets and timings.

## Formal measurement bounds and performance metrics (2026-10-23)

Formal `measure` now enforces P<=H before dataset access or output creation.
P>H remains available through the runtime and model/counting pipeline.
End-to-end measured rows report useful GFLOP/s, speedup against an explicit
same-layout P=1/H=1 serial measurement, and efficiency=speedup/H. Baseline
input/build/protocol/checksum mismatches are rejected; missing baselines and
model-only reports leave comparison fields blank.

- **507 Python/documentation tests passed**; strict MkDocs build passed.
- Tests cover early oversubscription rejection, real-H efficiency, original-n
  throughput under padding, missing/partial baselines, mismatched inputs,
  incompatible host/build/protocol, damaged reports and CLI baseline forwarding.
- Actual CLI validation used n=256/512/1024, seed 5206, group 0, range [-1,1].
  Serial P=1/H=1 and threaded P=4/H=4 each ran five warmups and fifteen repeats,
  with independent calibration and FP64 checks. Together they produced 630
  phase/end-to-end sample rows. All three performance formulas were checked
  against the saved measured medians.
- A P=16/H=4 measurement was rejected with no output directory. Offline model
  replay of the threaded traces succeeded without matrices and left measured
  performance columns blank.

Artifacts are retained locally under
`/var/folders/lw/j9_lmxtd6gx45dln0mlczmn80000gn/T/dmm-performance-q_1o1mci/`.
These runs verify the reporting pipeline and formulas; no fixed speedup is assumed.
The C++ runtime and teammate algorithm scope are unchanged.

## Endpoint-matched DSL and code organization (2026-10-23)

The migration based on main revision `d3733ad` replaces public tags/message IDs
with next-step endpoint matching and ordered payload components. `plan.hpp`
contains the DSL, `plan_build.cpp` validates it, and `plan_execute.cpp` interprets
it. `src/algorithms/` contains builders and selection; config is immutable and
builders return row bindings. The old Cannon facade and Algorithm concept are
removed. Python trace I/O/validation now lives in `dmm.trace`.

- Debug and Release: **77 C++ tests passed in each build**.
- Python and documentation: **488 tests passed**; strict MkDocs build passed.
- New protocol cases cover ordered multicomponent messages, strided views,
  source overwrite after send, shape/arity/adjacency checks, duplicate pairs,
  plan-scoped handles, operation order and stale superstep rejection.
- First/repeated allocation-free execution, memory peaks, tree counts, c=4
  redistribution, common lifecycle, FP64 correctness and failure/retry passed.
- Actual CLI smoke: n=17/65, P=4/16, serial, c=1, seed 5206, group 0, range [-1,1].
  Generation, model capture, independent calibration, five warmups and fifteen
  repetitions passed (four configurations, 420 phase/end-to-end sample rows).
- Offline model replay passed using new schema-2 measurement traces with a
  nonexistent input path. The previous real schema-1 CLI capture also replayed
  successfully without matrices. A committed pre-migration real trace fixture
  separately checks historical loading/counts without algorithm execution.

Artifacts are local temporary files under
`/var/folders/lw/j9_lmxtd6gx45dln0mlczmn80000gn/T/dmm-dsl-smoke-dkpxx9ro/`.
These verify integration, not performance rankings. Historical traces preserve
their recorded peaks/provenance; new records use ordered `regions` arrays.
Ring/2.5D and the full comparison remain assigned to teammates.

## Historical unified operation-plan runtime (2026-10-23)

The simplification based on main revision `c5ca662` removes the callback BSP
runtime and per-layout prepared classes. `build_cannon` declares operations;
`PreparedMatmul` owns the common lifecycle and `BspPlan` is the sole interpreter.
Validation was performed on the working tree before committing these changes.

- Debug and Release: **71 C++ tests passed in each build**.
- Python and documentation: **480 tests passed**; strict MkDocs build passed.
- Ownership, message validation, many-to-one traffic, memory peaks, trees and
  conversions now test the plan interpreter directly against independent counts.
- A separate one-node builder exercises the common lifecycle without Cannon.
- First/repeated no-allocation, input ownership, ordinary/prepared full-trace
  equality, shared-workspace concurrency and failure/retry checks passed.
- Actual CLI smoke: Cannon P=4/16 at original n=17/65, serial, seed 5206, group 0,
  FP32 range [-1,1]. Dataset generation, model capture, independent calibration,
  five warmups/fifteen repetitions and measurement-trace model replay passed.
  Four measured configurations produced 420 phase/end-to-end sample rows.

Smoke artifacts are local temporary files under
`/var/folders/lw/j9_lmxtd6gx45dln0mlczmn80000gn/T/dmm-plan-smoke-jvrydv8f/`.
They check functional integration, not performance rankings. The current
[contribution guide](contribute.md) provides the builder contract and reproducible
smoke profiles. Ring/2.5D and the full 24-configuration comparison remain pending.

## Historical framework run

Runtime source: `270300d9bc621ed977460ba5dc948c0debfd2f16`; captured with a clean working tree.
The later verification commit changes documentation and candidate labels/provenance in figures only.

This is a historical record of that revision. Cannon has since been unified:
ordinary calls and retained prepared calls now share one schedule and identical
common-input payload peaks. The old allocation-based capture remains valid for
its recorded source, but its memory values are not the current storage policy.
See the [contribution guide](contribute.md) for the unified interface.

## Functional checks

- Fresh Release build: **72 C++ tests passed**.
- Python and documentation: **473 tests passed**; MkDocs strict build passed.
- Allocation counters verify first/repeated prepared executions on caller/workers.
- Independent FP64 validation, coverage and saved-trace checks passed.
- Prepared binary trees match ordinary runtime outputs/FLOPs/messages/supersteps.
- Common prepared conversions cover P=64/256,c=4 without a 2.5D multiply implementation.

## Actual executions

Cannon common-route capture used n=256,512,1024 at P=16,64,256: nine actual
executions, 72 modeled scenarios and 144 budget cells with Cannon as the only
candidate. This is **not** a three-layout comparison or evidence that Cannon
beats the missing layouts. Selection figures name their available candidate set.

Prepared measurements used the same three sizes at P=1/H=1 and P=4/H=4.
Each configuration ran five warmups and fifteen validated repetitions (90
measured products in total). Independent micro calibrations were frozen before
each executor context. Saved prepared traces were also replayed offline with
a deliberately nonexistent data path, producing 24 model rows per context.

## Measured end-to-end results

Times below are seconds. Signed residual = (measured−predicted)/measured.

| P/H | n | Median | IQR | Frozen-model seconds | Relative residual | Peak words |
| --- | --- | --- | --- | --- | --- | --- |
| 1/1 | 256 | 0.00260754 | 0.0001765 | 0.00260667 | 0.03% | 196608 |
| 1/1 | 512 | 0.0219947 | 0.00235906 | 0.0208533 | 5.19% | 786432 |
| 1/1 | 1024 | 0.246111 | 0.049876 | 0.166826 | 32.22% | 3145728 |
| 4/4 | 256 | 0.00253171 | 0.00526158 | 0.00204758 | 19.12% | 131072 |
| 4/4 | 512 | 0.0496836 | 0.0746182 | 0.0068091 | 86.30% | 524288 |
| 4/4 | 1024 | 0.14263 | 0.0819514 | 0.0447178 | 68.65% | 2097152 |

The host reported eight physical cores, AppleClang 21.0 and unbound persistent
workers. Recorded load averages were approximately 46–47 during these runs.
Threaded IQRs are large and the largest residual is about 86%. This run verifies
pipeline execution and honest residual reporting; it does **not** establish stable
performance rankings or transferable coefficients. Scheduler/instrumentation
costs and cache effects are plausible contributors; these observations do not
isolate their causes. No matrix-specific refit was performed.

| Context | w seconds/FLOP | g seconds/word | l seconds/step |
| --- | --- | --- | --- |
| H=1 | 7.7684402e-11 | 1.1316681e-10 | 8.75e-09 |
| H=4 | 8.0307007e-11 | 1.6967392e-10 | 0.00015095875 |

## Reproduce and inspect

Raw artifacts and exact YAMLs are retained locally under
`results/framework_validation_nv8l0oxu/`. Large generated datasets/reports are
ignored by Git. Use `serial_final`, `threaded_final` and `model_cannon_final`
for the final reports; earlier exploratory outputs are retained separately.
The committed profiles and [protocol](measurement.md) reproduce
the method with a fresh output path; exact timings depend on the host.

```sh
cmake -S . -B build/fresh -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build/fresh --parallel
ctest --test-dir build/fresh --output-on-failure
uv run pytest -q tests/python doc/tests
uv run dmm measure --config configs/measure_serial.yaml
uv run dmm measure --config configs/measure_threaded.yaml
```

Dataset settings for this validation were seed=5206, group=0, sizes=256/512/1024,
FP32 range [0,255], with independently generated FP64 references. Every capture
and measurement completion manifest preserves source/build/settings and hashes.

## Pending teammate integration

Ring and 2.5D remain assigned to teammates. The full 24 actual configurations /
192 modeled scenarios have **not** been executed. `configs/model.yaml` and its
cardinality test define the target; missing implementations fail preflight.
No missing trace or measurement has been synthesized.

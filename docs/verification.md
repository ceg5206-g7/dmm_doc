# Framework verification record

## Unified operation-plan runtime (2026-10-23)

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

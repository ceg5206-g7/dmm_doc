# Prepared measurements and local calibration

```sh
uv run dmm measure --config configs/measure_serial.yaml
uv run dmm measure --config configs/measure_threaded.yaml
```

The profiles use existing group-0 matrices at n=256,512,1024. The serial profile
uses Cannon P=1/H=1; the threaded profile uses P=4/H=4 and refuses to start unless
at least four physical cores are detected. Use a fresh output directory. Neither
input datasets nor existing reports are overwritten. Missing teammate algorithms
fail preflight. Profiles accept per-case layout/P/c, one shared worker count,
group, sizes, warmups and repeats. Defaults are five warmups and fifteen repeats.

## Measurement boundary and statistics

Algorithms expose `prepare(A,B,*,P,c)` and a prepared object with
`run_into(output,executor)` and `trace()`. The C++ [prepared runtime](api/cpp.md#preallocated-execution)
preallocates node buffers, message caches, operation lists and trace metadata.
Cannon is the reference adapter; ring/2.5D implementations remain teammate work.

The **internal native timer** starts at common input row shards and stops at
common output row shards. It includes zeroing C, all layout conversion/packing,
message copies, shared GEMM and executor scheduling. Allocation, host input
reset/distribution, output assembly, Python dispatch, FP64 validation and writes
are outside. Every warmup and measured output is validated against the saved
FP64 oracle. C++ tests count allocations on caller and worker threads, including
the first execution. Do not combine these times with the convenience `bench`
CSV: that command intentionally includes wrappers and allocation.

End-to-end and six phases report total, compute, copy and scheduling/synchronization
seconds. Each of three blocking batches measures per-node task times; their
maximum is the local compute/copy component. The remaining wall time includes
scheduler and instrumentation overhead. If P>H, it also includes batching of
logical nodes. Fused receive-and-sum is counted as FLOPs but timed in the copy
bucket. Phase medians are independent statistics and need not add up to the
median whole execution. Native totals sum each repetition's phase times excluding
only input/output conversion before computing median/IQR. Linear quartiles use
position (N−1)*p. Inactive components can be zero.

## Independent calibration and held-out checks

Before any full matrix run, the same persistent executor measures:

| Parameter | Independent work | Units |
| --- | --- | --- |
| w | Shared GEMM, local blocks 32/64/96, 16 products per sample | Critical local seconds/FLOP |
| g | Two explicit copies (send plus receive), 1024/4096/16384 words, 128 iterations | Critical local seconds/word |
| l | Three empty blocking executor batches, 100 logical steps | Wall seconds/superstep |

Each micro configuration uses the same warmup/repeat counts as the matrix run.
Parameters are the median of normalized sample rates pooled equally across
sizes; IQR includes size/cache effects as well as run variation. Each parameter
set is fixed per H before the first matrix execution. Full sizes 32/64/96 are
rejected to keep them disjoint from the calibration matrix dimensions. No matrix
sample or per-case refit changes the coefficients. Measured data and residuals
remain visible even when the prediction is poor.

The runtime uses unbound persistent std::threads with static round-robin tasks.
This binding policy is the same for calibration and full runs. Metadata records
physical/logical CPUs, available affinity mask, compiler/build, OS, source
revision/dirty state and load averages before/after. Frequency, OS placement and
background work can still vary. These coefficients describe this local copied
message emulator; they do not measure a real distributed network.

## Saved artifacts

- `microbenchmarks.csv`: all raw calibration samples and their work/word/step units.
- `calibration.json`: frozen w/g/l in seconds and IQRs, H and exact methodology.
- `samples.csv`: every repetition's end-to-end/six-phase timing components.
- `measured_model.csv`: actual trace counts/peaks, median/IQR components, measured
  median, normalized `T_model`, predicted seconds and signed/relative residuals.
- `correctness.csv`: maximum absolute/Frobenius relative error per measured repeat.
- `traces/`: executed prepared traces and complete, checksummed capture manifest.
- `heldout_residuals.png`: measured vs predicted end-to-end costs and signed residuals.
- `manifest.json`: completion marker, settings, provenance and artifact checksums.

`T_model = W_crit + (g/w)*C_crit + (l/w)*S` is in FLOP-equivalents;
`T_model_seconds = w*T_model` and `t_measured` are in seconds.
The signed residual is measured minus predicted, and the relative residual uses
measured time as denominator. Packing, local copies not charged as communication,
cache effects and scheduler variation can explain residuals. These effects are
not removed by fitting each matrix separately. A failed/partial directory has
no completion manifest and does not establish a completed experiment.

Prepared message caches remain allocated and counted while idle, so their peaks
are now identical for ordinary and retained Cannon calls using the same inputs
and configuration. Earlier allocation-based traces can differ; keep their source
revision and use actual captured peaks for predictions and feasibility. Prepared traces can be passed directly to
`dmm model --traces <measurement_output>/traces` with matching cases/sizes/group/workers
and a fresh model output path. That recomputes budget scans/charts without
reloading matrices or rerunning algorithms. Measured medians remain in the
measurement report; hypothetical gamma/lambda sweeps keep t_measured empty.

## Remaining integration gate

Cannon validates the shared pipeline. The complete 24-configuration/192-scenario
comparison still needs actual ring and 2.5D algorithms from teammates; neither
these measurements nor cardinality tests claim that integration has passed.

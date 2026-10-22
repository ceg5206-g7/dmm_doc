# Experiment Methodology

## Timing boundary

Python uses `time.perf_counter()` around `matmul(A, B, executor)`.
Google Benchmark is not used.

Included work:

- Python wrapper and binding dispatch, input validation and copies into C++.
- Common row-shard construction, packing, row/block conversion and tile initialization.
- Message payload copies and delivery between supersteps.
- Executor dispatch per superstep, shared blocked GEMM, completion waits and tile assembly.
- Copying the native result into its independent NumPy output.

Dataset/reference loading, pool construction and shutdown, warmups, correctness
validation and output writes are outside the measured interval. The GIL is
released for native computation and output copying. Small matrices can be
dominated by conversion and dispatch overhead.

For each case, all configured warmups run first without recording time. Each
repeat then produces one measured result, which must validate before its row is
written and flushed. Non-positive or non-finite elapsed times are rejected.
The runner has no automatic iteration calibration.

Settings iterate in YAML mode order and worker-list order. Within each setting,
the runner visits algorithm, group and size combinations, then repetitions.
Executors persist across cases; inputs persist across a case's warmups/repeats.

## Correctness and arithmetic

A/B and computed C use FP32. The saved reference is computed independently by
NumPy after converting the stored A/B values to FP64. Native tests separately
use scalar FP64 accumulation into `Matrix<double>`.

Each result element must satisfy

$$
|C_{ij}-C^{ref}_{ij}| \leq 10^{-4} + 10^{-3}|C^{ref}_{ij}|.
$$

Non-finite outputs fail validation. Maximum absolute error and Frobenius relative
error are computed in FP64; when the reference norm is zero, the reported norm
error is the absolute error norm. Tolerances accommodate FP32 rounding; they
do not establish bitwise equality to a different accumulation order. Cannon's
stage ordering is tested explicitly with cancellation-sensitive inputs.

## Aggregation and speedup

The default and full convenience profile warm up 5 times and measure 15 repeats.
Smoke/lite profiles override these counts for fast pipeline checks. Each input
group reports the sample median and IQR=Q75−Q25, using linear interpolation at
position (N−1)*p in sorted samples. For multiple input groups, summary median_time_s
is the median of group medians; iqr_time_s is the median of group IQRs. This
preserves repeat variability without treating different inputs as repetitions.
A single sample has IQR zero; it does not demonstrate stable performance.

For each matching layout/P/c/size, speedup is the serial summary median divided
by the threaded summary median. Both must contain matching input groups.
Duplicate samples, incomplete/inconsistent repeat sets, missing baselines and
invalid times are rejected. Legacy CSVs without repeat count as one sample.
Sequential summary speedup is one. Runtime plots use medians. The timing boundary
still includes allocation until the separate preallocated driver is implemented.

Do not combine runs with different timing boundaries, builds or input settings.
Earlier Google Benchmark or persistent-runner timings are not directly comparable.

## Reproducibility and limits

Keep the CSV, companion JSON, dataset manifest and corresponding source revision
together. A dirty Git flag records that local changes exist; it does not preserve
those changes. Record machine load and experimental conditions when interpreting
results. Warmup and repetition reduce some variability but do not guarantee
stable frequency, cache state or absence of scheduling noise.

The logical grid runs P nodes per superstep regardless of worker count.
Workers share one process and machine. Neighbour shifts are copied messages
between node-local buffers, with a barrier after each superstep, but no sleep
or delay models network time. Measured thread speedup describes this
implementation only. Communication costs come from the
[BSP counts](design.md#communication-model) and chosen $g$/$l$ parameters,
not from wall-clock. Benchmark CSVs do not yet persist traces; use
`matmul_traced` for the complete common path and `count_native(trace)` for
the body excluding only input/output conversion. Trace records per-node peak payload words; these differ from whole-program
resident memory and measured DRAM traffic.

Partial CSVs preserve completed measurements after failure. They do not establish
that the requested experiment finished. Analysis validation checks consistency,
not completion against an unavailable original execution plan.

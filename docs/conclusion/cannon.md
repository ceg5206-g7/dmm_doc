# Cannon: conclusions against the proposal

Cannon provides a validated 2D baseline for the proposal's layout comparison.
Its executed counts expose a trade-off between local work, critical communication,
synchronization and memory. The limited single-machine measurements show higher
parallel efficiency at larger matrix sizes in this run. Whether replication
improves on this baseline remains open until ring and 2.5D are integrated.

## Scope and evidence

This case addresses the research question in proposal section 1, the common
input/output and memory rules in section 3, and the analysis and interpretation
requirements in sections 4–6. See the [proposal](../proposal.md).

Evidence comes from the clean-source run at
`bd85893db2a75a67b4504cd952e30e3546632097` on 2026-10-23:

- Nine main captures: n=256/512/1024, P=16/64/256, c=1. All have N=n.
- Nine padding captures: n=17/65/257 with the same logical grids.
- The main captures yield 72 gamma/lambda scenarios and 144 budget checks.
- Formal measurements use P=1/H=1 and P=4/H=4, five warmups and fifteen repeats,
  on a host with eight physical cores. P>H captures supply correctness and counts.
- Inputs use seed 5206, group 0, FP32 range [0,255]. FP64 checks, exact block
  coverage and independent count checks passed. The eighteen main/padding
  captures have Frobenius relative errors between approximately 6.05e-8 and
  4.27e-7, with the elementwise atol=1e-4, rtol=1e-3 rule also satisfied.

The [verification record](../verification.md#complete-cannon-pipeline-2026-10-23)
preserves the build, full pipeline, statistics and artifact inventory. Raw
artifacts are local and Git-ignored under
`results/cannon_pipeline_20261023__h524qgo/`; the tables below retain the evidence
needed to read this conclusion without those files.

## Communication and synchronization trade-off

Write N=ceil(n/P)*P, q=sqrt(P), b=N/q and h=N/P. For the current schedule at P>1,
the independently checked native-body counts are:

```text
F_total = 2*N^3
W_crit  = 2*N^3/P
C_crit  = 2*q*b^2 = 2*N^2/sqrt(P)
C_total = 2*(q-1)*(q+P)*b^2
S       = q+1
```

This establishes the proposal's 2D critical communication order for this
implementation, including initial skew and inter-round shifts. It does not
establish proximity to a communication lower bound or validate a real network.
Global total communication and critical communication have different roles;
the BSP time model uses C_crit.

The captured common-input/output counts at n=1024 illustrate the trade-off:

| P | W_crit (FLOPs) | C_crit (words) | C_total (words) | S | Peak words/node |
| --- | --- | --- | --- | --- | --- |
| 16 | 134217728 | 671744 | 10223616 | 11 | 524288 |
| 64 | 33554432 | 305152 | 19267584 | 15 | 131072 |
| 256 | 8388608 | 142592 | 36372480 | 23 | 32768 |

Across these grids, increasing P lowers per-node work, critical communication
and peak storage, while total communicated words and supersteps rise. Thus the
cost must retain separate computation, bandwidth and synchronization terms.

The sweep demonstrates why gamma and lambda are separate parameters. At
n=256, P=256 and lambda=10000, synchronization contributes 62.16% of normalized
end-to-end cost when gamma=1. With gamma=1000 at the same n/P/lambda,
communication contributes 96.11%. These are model decompositions under the
listed BSP assumptions; they are not measured copy/barrier time fractions.

## Cost of common input/output

The common row-to-row route adds A conversion, B conversion and C conversion
to the native body. For P>1 their combined cost is:

```text
Delta W_crit  = 0
Delta C_crit  = 3*(q-1)*h*b
Delta C_total = 3*P*(q-1)*h*b
Delta S       = 6
Delta T_model = gamma*3*(q-1)*h*b + 6*lambda
```

Conversion increases critical communicated words over the native body by
28.13%, 16.41% and 8.79% at P=16/64/256 respectively. At P=16, six of the eleven
end-to-end supersteps belong to conversion. Step counts are not time fractions:
their importance depends on n, gamma and lambda.

The implication for the proposal is that a native-layout communication saving
must be evaluated again after adding conversion. A comparison that omits these
phases would change the question being answered. This run establishes Cannon's
conversion baseline; a change in the winning layout requires other candidates.
Native counts/times are filtered from the same execution, rather than obtained
from a separate native-input implementation.

## Memory feasibility and padding

All nine main captures record peak_words=8*N²/P under the current buffer and
message-cache policy. They fit both proposal budgets, but **exactly reach the
mu=8 cap**, leaving no spare logical payload capacity. This factor is observed
for these captures, not a minimum-storage theorem for Cannon. Peaks include
owned payloads, scratch and allocated message caches even while idle; they do
not describe whole-process RSS or DRAM traffic.

Padding changes feasibility because actual work/storage use N while the budget
remains M_cap=mu*n_original²/P. In the padding captures all nine configurations
exceed mu=8. At mu=32 the outcomes are:

| Original n | P=16 | P=64 | P=256 |
| --- | --- | --- | --- |
| 17 | Feasible (N=32) | Infeasible (N=64) | Infeasible (N=256) |
| 65 | Feasible (N=80) | Feasible (N=128) | Infeasible (N=256) |
| 257 | Feasible (N=272) | Feasible (N=320) | Feasible (N=512) |

Together these give twelve infeasible and six feasible size/P/budget checks.
All nine configurations still execute correctly; infeasibility is a budget
decision. Supporting arbitrary positive n therefore does not imply every n/P
combination is an economical configuration. Keep padded counts and original-n
useful throughput separate.

## Single-machine validation and model limits

The following end-to-end times include necessary packing, layout conversion,
message copies, C initialization, GEMM and scheduling. Allocation, host adapters,
dataset loading and FP64 validation are outside the internal timer. Speedup uses
the completed same-input P=1/H=1 shared-kernel measurement as its baseline.

| n | P=4/H=4 median / IQR (ms) | GFLOP/s | Speedup | Efficiency | Relative residual |
| --- | --- | --- | --- | --- | --- |
| 256 | 1.184209 / 0.162230 | 28.33 | 2.002× | 50.05% | 13.77% |
| 512 | 5.716000 / 0.152082 | 46.96 | 3.333× | 83.33% | 2.86% |
| 1024 | 41.657750 / 0.815145 | 51.55 | 4.040× | 101.00% | 0.16% |

Efficiency is speedup divided by real H=4. It is not clamped: 101.00% may be
affected by caches and measurement variation; their contributions have not been
isolated. The observed improvement with n is consistent with cubic computation
growing faster than quadratic data movement, with a fixed step count at P=4.
Three sizes and one fixed-seed input group do not establish a universal trend.

Independent GEMM, copy and empty-superstep microbenchmarks froze w/g/l before
each executor context's held-out matrix measurements. The total residual gets
smaller with n in the threaded run, but total agreement does not establish
phase accuracy. At n=1024, input/output conversion residuals are 69.76% and
69.46%, while the multiply phase residual is −1.63%. Significant phase errors
coexist with the 0.16% end-to-end residual. Phase medians are independent and
must not be summed to reconstruct the end-to-end median.

These results support limited applicability of the frozen local model to this
shared-memory implementation. Local packing/copy costs, instrumentation,
scheduling and cache variation remain possible sources of residuals; no
per-matrix refit removes them. See the [measurement protocol](../measurement.md)
for definitions and timing boundaries. The P=16/64/256 captures are not physical
parallel performance measurements, and memcpy coefficients are not network
bandwidth measurements.

## Questions awaiting the other layouts

Cannon alone cannot determine whether replication saves enough communication
to pay for replication, reduction and conversion, where a layout transition
occurs as gamma/lambda change, or whether every layout has a winning region.
Its current selection chart has one candidate and establishes feasibility only.

After teammate integration, compare layouts at the same original n, P, gamma,
lambda and memory cap. Exclude infeasible candidates first, then compare both
native and common-route costs, retaining ties. For a feasible replicated
candidate R to beat feasible Cannon C on the common route, it must satisfy:

```text
gamma*(C_crit_C - C_crit_R)
    > (W_crit_R - W_crit_C) + lambda*(S_R - S_C)
```

R's counts must include reduction work, tree communication and common
conversions. Filling those terms with actual 2.5D traces is the remaining
evidence needed to answer the proposal's replication question. Ring and 2.5D,
including c=1 degeneration, and the full 24-execution/192-scenario comparison
remain pending. No multi-machine speedup or three-layout winner is claimed.

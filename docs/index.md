# Distributed Matrix Multiplication

CEG5206 experiments on matrix layouts and communication costs. C++20 implements
Ring, Cannon, 2.5D and task execution. The pybind11 extension connects them to Python for
datasets, warmup, repeated measurements, analysis and plots.

Inputs and outputs use FP32. Saved reference products independently accumulate
in FP64. The three layouts run on one machine with a shared BSP runtime.
The runtime copies payloads, records per-superstep communication and tracks
peak node payload memory. All layouts accept common row-sharded inputs and
return common row-sharded outputs.

| Guide | Contents |
| --- | --- |
| [Usage and configuration](usage.md) | Installation, commands, YAML, datasets and Python examples |
| [System design](design.md) | Python/C++ responsibilities, BSP communication model, binding and executors |
| [Algorithms](algorithms/overview.md) | Layouts, supersteps, message routes and counts |
| [Experiment methodology](experiments.md) | Timing, validation, averaging and reproducibility |
| [Results and Discussion](results.md) | Formal comparison of communication, budgets and single-machine measurements |
| [Cannon case study](conclusion/cannon.md) | Historical Cannon-only verification and interpretation |
| [Python API](api/index.md) | Module pages generated from the current package |
| [C++ API](api/cpp.md) | Matrix templates, BSP runtime, traces and costs, algorithms and executors |
| [Adding an algorithm](contribute.md) | Implementation, integration and verification |

Commands and code paths are relative to the
[dmm checkout](https://github.com/ceg5206-g7/dmm) unless stated otherwise.
This documentation lives in [dmm_doc](https://github.com/ceg5206-g7/dmm_doc),
mounted at `doc/` in the main checkout.

All layouts share a blocked FP32 kernel with fixed tile 32.
See the [framework implementation contract](framework.md) for verification milestones.

# Distributed Matrix Multiplication

CEG5206 experiments on matrix layouts and communication costs. C++20 implements
Cannon and task execution; pybind11 connects them to a Python framework for
datasets, warmup, repeated measurements, analysis and plots.

Inputs and outputs use FP32. Saved reference products independently accumulate
in FP64. Cannon runs on one machine as P=q² logical nodes of a configurable square grid on
a BSP runtime: block skews and shifts are real copied messages, counted per
superstep but never delayed. BSP costs are computed from those counts afterwards.
Local and message payload memory peaks are tracked per node. 1D ring, replicated
2.5D remain future work. Common row-shard conversion and padding are implemented.

| Guide | Contents |
| --- | --- |
| [Usage and configuration](usage.md) | Installation, commands, YAML, datasets and Python examples |
| [System design](design.md) | Python/C++ responsibilities, BSP communication model, binding and executors |
| [Algorithms](algorithms/overview.md) | Availability, Cannon's supersteps, message routes and counts |
| [Experiment methodology](experiments.md) | Timing, validation, averaging and reproducibility |
| [Case conclusions: Cannon](conclusion/cannon.md) | Answers to the proposal's cost, memory, conversion and model questions |
| [Python API](api/index.md) | Module pages generated from the current package |
| [C++ API](api/cpp.md) | Matrix templates, BSP runtime, traces and costs, algorithms and executors |
| [Adding an algorithm](contribute.md) | Implementation, integration and verification |

Commands and code paths are relative to the
[dmm checkout](https://github.com/ceg5206-g7/dmm) unless stated otherwise.
This documentation lives in [dmm_doc](https://github.com/ceg5206-g7/dmm_doc),
mounted at `doc/` in the main checkout.

All layouts share a blocked FP32 kernel with fixed tile 32.
See the [framework implementation contract](framework.md) for remaining milestones.

# Distributed Matrix Multiplication

CEG5206 experiments on matrix layouts and communication costs. C++20 implements
Cannon and task execution; pybind11 connects them to a Python framework for
datasets, warmup, repeated measurements, analysis and plots.

Inputs and outputs use FP32. Saved reference products independently accumulate
in FP64. Cannon currently runs on one machine with a fixed logical 4 × 4 block
grid. Communication simulation, 1D ring and replicated 2.5D implementations are
future work; the current measurements describe local computation and dispatch.

| Guide | Contents |
| --- | --- |
| [Usage and configuration](usage.md) | Installation, commands, YAML, datasets and Python examples |
| [System design](design.md) | Python/C++ responsibilities, binding, ownership and executor lifecycle |
| [Algorithms](algorithms/overview.md) | Availability and Cannon's work decomposition |
| [Experiment methodology](experiments.md) | Timing, validation, averaging and reproducibility |
| [Python API](api/index.md) | Module pages generated from the current package |
| [C++ API](api/cpp.md) | Matrix templates, block views, algorithms and executors |
| [Adding an algorithm](contribute.md) | Implementation, integration and verification |

Commands and code paths are relative to the
[dmm checkout](https://github.com/ceg5206-g7/dmm) unless stated otherwise.
This documentation lives in [dmm_doc](https://github.com/ceg5206-g7/dmm_doc),
mounted at `doc/` in the main checkout.

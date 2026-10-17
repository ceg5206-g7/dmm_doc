# Adding an Algorithm

Keep computation in C++ and orchestration in Python. Cannon is the existing
example; its fixed grid is an implementation detail rather than a requirement
for every algorithm.

## Files to change

| File | Change |
| --- | --- |
| `include/dmm/<name>.hpp`, `src/<name>.cpp` | Implement a stateless native algorithm and assert its concept |
| `CMakeLists.txt` | Add the source to `dmm_core` and its native test to `dmm_tests` |
| `src/bindings.cpp` | Add the NumPy/native entry point with validation and ownership handling |
| `python/dmm/algorithms/<name>.py` | Expose size validation and multiplication |
| `python/dmm/config.py` | Add a name only if it is not already recognized |
| `tests/cpp/test_<name>.cpp` | Check arithmetic, decomposition, invalid inputs and executors |
| `tests/python/test_<name>.py` | Check binding types, strides, input preservation and output ownership |
| `configs/benchmark_<name>.yaml` | Select supported sizes and a fresh output path |
| `doc/docs/algorithms/`, `doc/mkdocs.yml` | Document availability, arithmetic, constraints and measurement limits |

`fox`, `strassen` and `cw` are already recognized but unimplemented. Their names
alone do not make them runnable. The runner dynamically imports the selected
module; no additional CLI branch is needed.

## Native and Python contracts

Use the native signature from the [C++ reference](api/cpp.md#algorithms).
Borrow the executor for each call and return independent FP32 matrix storage.
Use `Executor::for_each` for tasks without creating a nested pool. Preserve
inputs and keep task-referenced storage alive until the synchronous batch ends.
Algorithm implementations need no inheritance hierarchy.

The Python module exposes:

```python
def validate_size(size: int) -> None: ...
def matmul(A: np.ndarray, B: np.ndarray, executor: Executor) -> np.ndarray: ...
```

Size validation must reject unsupported sizes without allocating matrices or
workers. The runner checks every requested size before creating output. Direct
binding calls still need complete type, shape, dtype and finiteness validation.
Support read-only and strided inputs or document and test any explicit limits.

Keep timing, file I/O, dataset generation and reference computation out of the
algorithm. Python owns those operations. Ordinary task exceptions must propagate
through the supplied executor.

## Verification

Cover several supported sizes, invalid inputs, zero/identity/random matrices,
independent FP64 reference comparison, input preservation and repeated calls.
Run with serial execution and one/multiple worker threads. Test algorithm-specific
invariants; Cannon checks cancellation-sensitive stage ordering. Check binding
output dtype and independence, plus preflight rejection before experiment setup.

From the main checkout:

```sh
uv sync --locked --reinstall-package dmm
uv run --locked pytest -q
uv run cmake --preset debug
uv run cmake --build --preset debug --parallel
ctest --preset debug --no-tests=error
uv run cmake --preset release
uv run cmake --build --preset release --parallel
ctest --preset release --no-tests=error
uv run --locked pytest doc/tests -q
cd doc
uvx --with-requirements requirements.txt mkdocs build --strict
```

Run a small experiment using a fresh output path before comparing performance.
Record the actual build and environment; correctness tests do not establish
speedup. Commit documentation changes in `dmm_doc` before updating the parent
project's submodule pointer.

## Writing diagrams and equations

Use fenced `mermaid` blocks with `accTitle` and `accDescr` for accessible diagrams.
Use `$...$` for inline math and `$$` on separate lines for display equations.
The [system design](design.md#modules-and-data-flow) and
[Cannon guide](algorithms/cannon.md#cannons-logical-block-grid) provide examples.
Pinned Mermaid and MathJax assets render these features in the browser.

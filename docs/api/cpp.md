# C++ API

Headers live in `include/dmm/`. This page describes the native interface;
the [Python reference](index.md) is generated separately with pydoc.

## Matrices and block views

```cpp
template <typename T>
struct Matrix {
    explicit Matrix(std::size_t size);
    std::size_t size;
    std::vector<T> values;
    MatrixBlockView<T> block(std::size_t row, std::size_t col,
                            std::size_t block_size) const;
};

template <typename T>
struct MatrixBlockView {
    const T* data;
    std::size_t size;
    std::size_t stride;
    const T& operator()(std::size_t row, std::size_t col) const noexcept;
};
```

Matrices own zero-initialized row-major square storage. Construction rejects
zero dimensions and size arithmetic overflow using `sizeof(T)`. `block` rejects
malformed storage and empty/out-of-bounds blocks. Views are read-only, trivially
copyable borrowers; keep the source alive and its storage stable.

Cannon uses `Matrix<float>` and `MatrixBlockView<float>`. Independent native test
references use `Matrix<double>`. Template definitions are in `algorithm.hpp`.

## Algorithms

```cpp
struct Cannon {
    Matrix<float> matmul(const Matrix<float>& a, const Matrix<float>& b,
                         Executor& executor) const;
};
static_assert(Algorithm<Cannon>);
```

`Algorithm<T>` is a C++20 concept requiring a const algorithm to accept two
FP32 matrices plus an executor reference and return exactly `Matrix<float>`
by value. Algorithms use plain structs without a virtual base. Cannon validates
matching square storage and dimensions divisible by four.

## Executors

```cpp
class Executor {
public:
    virtual ~Executor() = default;
    virtual void for_each(std::size_t count,
                         const std::function<void(std::size_t)>& task) = 0;
};

// SerialExecutor and ThreadExecutor implement Executor.
std::unique_ptr<Executor> make_executor(std::size_t workers);
```

`count` is the number of tasks; `task(index)` performs task `index` for indices
from zero through `count-1`. The call is synchronous and propagates task errors.
Cannon submits 16 tasks in a single batch. A zero task count does no work.

`make_executor(0)` creates serial execution. Positive counts create a thread
pool. `ThreadExecutor` starts workers at construction, provides `enter()` and
idempotent `close()`, and closes on destruction. Calls sharing a pool serialize
batches; closing waits for the active batch. See [lifecycle rules](../design.md#executors).

## Native correctness validation

```cpp
struct Error {
    double max_absolute;
    double frobenius_relative;
};

Error validate(const Matrix<float>& result, const Matrix<double>& reference,
               double atol = 1e-4, double rtol = 1e-3);
```

Validation checks dimensions and both storage lengths, finite tolerances and
values, and elementwise absolute/relative tolerance. Malformed arguments throw
`std::invalid_argument`; incorrect/non-finite values throw `std::runtime_error`.
The benchmark uses its Python NumPy validator; native tests use this function.

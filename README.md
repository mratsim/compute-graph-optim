# Compute Graph Optimisations

A repo to test compile-time and runtime computation graph optimisation.

The main goal is to implement an alternative operation fusion scheme for [Laser](https://github.com/numforge/laser)
and also research into specialized compiler frontend for HPC and machine learning.

Currently Laser offers operation fusion in the following form:

```Nim
# x, y and z are tensors, .+ and .* are the elementwise addition and multiplication
# x = (x .+ y .* z) .* cos(y .+ z)
forEach x in a, y in b, z in c:
  x += y * z
  let temp = cos(y + z)
  x *= temp
```

This is very flexible however there are a couple of shortcomings:

1. Readability

Something like the following would be more readable

```Nim
x = fuse: (x + y * z) * cos(y + z)
```

2. Composition

There is no possibility to compose 2 forEach loops without introducing an intermediate result.

3. Runtime SIMD usage

The code will use SIMD (if #pragma omp simd) but only for the lowest common denominator of the target platform.
For example SSE2 on x86_64.

4. Custom optimized function implementation

Related to point 2 and 3,

Laser has very efficient implementations of function like the exponential which cannot be used in the current `forEach` loop.

## Approaches

### Overloading

3 and 4 could be solved by `forEach` implementing type specialization for the target platform, for example on x86_64 using:
  - float32
  - m128
  - m256
  - m512

And then relying on overloading using the [following vectorization template](https://github.com/numforge/laser/blob/e898f027e6d08d90542714234629da984d4e4639/benchmarks/vector_math/bench_exp.nim#L105-L153):

```Nim
func round_down_power_of_2(x: Natural, step: static Natural): int {.inline.} =
  static: assert (step and (step - 1)) == 0, "Step must be a power of 2"
  result = x and not(step - 1)

template vectorize(
      wrapped_func,
      funcname,
      simd_load,
      simd_store: untyped,
      unroll_factor: int) =
  proc funcname(dst, src: ptr UncheckedArray[float32], len: Natural) =
    let unroll_stop = len.round_down_power_of_2(unroll_factor)

    for i in countup(0, unroll_stop - 1, unroll_factor):
      dst[i].addr.simd_store src[i].addr.simd_load.wrapped_func
    for i in unroll_stop ..< len:
      dst[i] = src[i]
```

This would requires tracking down each variables checking if they come from a tensor or are broadcasted,
to properly SIMDize them.

### Compiler / Compute Graph

This would allow composition of building blocks into a computation graph in an approach similar to [what was discussed here](https://github.com/mratsim/Arraymancer/issues/347).

The main challenge is solving the Expression Problem.

# Upgrade on example 8 and 9

# Same logic but we extend that to Nim sequences
# and add dynamic dispatch depending on SIMD supported:
#   Generic, SSE or AVX

import macros

# ###########################
#
#         SIMD setup
#
# ###########################

when defined(vcc):
  {.pragma: x86_type, byCopy, header:"<intrin.h>".}
  {.pragma: x86, noDecl, header:"<intrin.h>".}
else:
  {.pragma: x86_type, byCopy, header:"<x86intrin.h>".}
  {.pragma: x86, noDecl, header:"<x86intrin.h>".}

type
  m128* {.importc: "__m128", x86_type.} = object
    raw: array[4, float32]
  m256* {.importc: "__m256", x86_type.} = object
    raw: array[8, float32]

type
  SimdSupported = enum
    Sse
    Avx
    AvxFma

  SimdOp = enum
    simdSetZero
    simdBroadcast
    simdLoadA
    simdLoadU
    simdStoreA
    simdStoreU
    simdAdd
    simdMul
    simdFma

func mm_setzero_ps(): m128 {.importc: "_mm_setzero_ps", x86.}
func mm_set1_ps(a: float32): m128 {.importc: "_mm_set1_ps", x86.}
func mm_load_ps(aligned_mem_addr: ptr float32): m128 {.importc: "_mm_load_ps", x86.}
func mm_loadu_ps(data: ptr float32): m128 {.importc: "_mm_loadu_ps", x86.}
func mm_store_ps(mem_addr: ptr float32, a: m128) {.importc: "_mm_store_ps", x86.}
func mm_storeu_ps(mem_addr: ptr float32, a: m128) {.importc: "_mm_storeu_ps", x86.}
func mm_add_ps(a, b: m128): m128 {.importc: "_mm_add_ps", x86.}
func mm_sub_ps(a, b: m128): m128 {.importc: "_mm_sub_ps", x86.}
func mm_mul_ps(a, b: m128): m128 {.importc: "_mm_mul_ps", x86.}

template sse_fma_fallback(a, b, c: m128): m128 =
  mm_add_ps(mm_mul_ps(a, b), c)

func mm256_setzero_ps(): m256 {.importc: "_mm256_setzero_ps", x86.}
func mm256_set1_ps(a: float32): m256 {.importc: "_mm256_set1_ps", x86.}
func mm256_load_ps(aligned_mem_addr: ptr float32): m256 {.importc: "_mm256_load_ps", x86.}
func mm256_loadu_ps(mem_addr: ptr float32): m256 {.importc: "_mm256_loadu_ps", x86.}
func mm256_store_ps(mem_addr: ptr float32, a: m256) {.importc: "_mm256_store_ps", x86.}
func mm256_storeu_ps(mem_addr: ptr float32, a: m256) {.importc: "_mm256_storeu_ps", x86.}
func mm256_add_ps(a, b: m256): m256 {.importc: "_mm256_add_ps", x86.}
func mm256_mul_ps(a, b: m256): m256 {.importc: "_mm256_mul_ps", x86.}
func mm256_sub_ps(a, b: m256): m256 {.importc: "_mm256_sub_ps", x86.}

func mm256_fmadd_ps(a, b, c: m256): m256 {.importc: "_mm256_fmadd_ps", x86.}
template avx_fma_fallback(a, b, c: m128): m128 =
  mm256_add_ps(mm256_mul_ps(a, b), c)

proc genSimdTableX86(): array[SimdSupported, array[SimdOp, NimNode]] =

  let sse: array[SimdOp, NimNode] = [
    simdSetZero:   newIdentNode"mm_setzero_ps",
    simdBroadcast: newIdentNode"mm_set1_ps",
    simdLoadA:     newIdentNode"mm_load_ps",
    simdLoadU:     newIdentNode"mm_loadu_ps",
    simdStoreA:    newIdentNode"mm_store_ps",
    simdStoreU:    newIdentNode"mm_storeu_ps",
    simdAdd:       newIdentNode"mm_add_ps",
    simdMul:       newIdentNode"mm_mul_ps",
    simdFma:       newIdentNode"sse_fma_fallback"
  ]

  let avx: array[SimdOp, NimNode] = [
    simdSetZero:   newIdentNode"mm256_setzero_ps",
    simdBroadcast: newIdentNode"mm256_set1_ps",
    simdLoadA:     newIdentNode"mm256_load_ps",
    simdLoadU:     newIdentNode"mm256_loadu_ps",
    simdStoreA:    newIdentNode"mm256_store_ps",
    simdStoreU:    newIdentNode"mm256_storeu_ps",
    simdAdd:       newIdentNode"mm256_add_ps",
    simdMul:       newIdentNode"mm256_mul_ps",
    simdFma:       newIdentNode"avx_fma_fallback"
  ]

  var avx_fma = avx
  avx_fma[simdFma] = newIdentNode"mm256_fmadd_ps"

  result = [
    Sse: sse,
    Avx: avx,
    AvxFma: avx_fma
  ]

let SimdTable{.compileTime.} = genSimdTableX86()

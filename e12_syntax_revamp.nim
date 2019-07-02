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
  SimdArch = enum
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
    simdType

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

proc genSimdTableX86(): array[SimdArch, array[SimdOp, NimNode]] =

  let sse: array[SimdOp, NimNode] = [
    simdSetZero:   newIdentNode"mm_setzero_ps",
    simdBroadcast: newIdentNode"mm_set1_ps",
    simdLoadA:     newIdentNode"mm_load_ps",
    simdLoadU:     newIdentNode"mm_loadu_ps",
    simdStoreA:    newIdentNode"mm_store_ps",
    simdStoreU:    newIdentNode"mm_storeu_ps",
    simdAdd:       newIdentNode"mm_add_ps",
    simdMul:       newIdentNode"mm_mul_ps",
    simdFma:       newIdentNode"sse_fma_fallback",
    simdType:      newIdentNode"m128"
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
    simdFma:       newIdentNode"avx_fma_fallback",
    simdType:      newIdentNode"m256"
  ]

  var avx_fma = avx
  avx_fma[simdFma] = newIdentNode"mm256_fmadd_ps"

  result = [
    Sse: sse,
    Avx: avx,
    AvxFma: avx_fma
  ]

let SimdTable{.compileTime.} = genSimdTableX86()

# ###########################################
#
#                Function Vectorizer
#
# ###########################################

func round_down_power_of_2(x: Natural, step: static Natural): int {.inline.} =
  static: assert (step and (step - 1)) == 0, "Step must be a power of 2"
  result = x and not(step - 1)

# ###########################################
#
#         Internal Graph Representation
#
# ###########################################
import macros

# For implementation ease, controlling side-effects and parallelism
# We want a functional side-effect free language
# However in-place updates like "+=" are just too ergonomic to pass up
# we simulate those in the AST via a persistent data-structure

import hashes, random, tables
var astNodeRng {.compileTime.} = initRand(0x42)
  ## Workaround for having no UUID for AstNodes
  ## at compile-time - https://github.com/nim-lang/RFCs/issues/131

type
  AstNodeKind = enum
    ## Elemental Op Kind

    # Expressions
    IntScalar   # Mark an integer scalar that will be broadcasted
    FloatScalar # Mark a float scalar that will be broadcasted
    Add         # Elementwise add
    Mul         # Elementwise mul

    # Symbols
    Input       # Input tensor node
    Output      # Mutable output tensor node
    LVal        # Temporary allocated node
    Assign      # Assignment statement

  AstNode = ref object
    case kind: AstNodeKind
    of Input:
      symId: int
    of Output, LVal:
      symLval: string
      version: int
      prev_version: AstNode      # Persistent data structure
    of IntScalar:
      intVal: int                # type to use? int64?
    of FloatScalar:
      floatVal: float
    of Assign, Add, Mul:
      lhs, rhs: AstNode

    ctHash: Hash                 # Compile-Time only Hash

  Coord = object

proc genHash(): Hash =
  Hash astNodeRng.rand(high(int))

proc hash(x: AstNode): Hash {.inline.} =
  when nimvm:
    x.cthash
  else: # Take its address
    cast[Hash](x)

proc input*(id: int): AstNode =
  when nimvm:
    AstNode(ctHash: genHash(), kind: Input, symId: id)
  else:
    AstNode(kind: Input, symId: id)

proc `+`*(a, b: AstNode): AstNode =
  when nimvm:
    AstNode(ctHash: genHash(), kind: Add, lhs: a, rhs: b)
  else:
    AstNode(kind: Add, lhs: a, rhs: b)

proc `*`*(a, b: AstNode): AstNode =
  when nimvm:
    AstNode(ctHash: genHash(), kind: Mul, lhs: a, rhs: b)
  else:
    AstNode(ctHash: genHash(), kind: Mul, lhs: a, rhs: b)

proc `*`*(a: AstNode, b: SomeInteger): AstNode =
  when nimvm:
    AstNode(
        ctHash: genHash(),
        kind: Mul,
        lhs: a,
        rhs: AstNode(kind: IntScalar, intVal: b)
      )
  else:
    AstNode(
        kind: Mul,
        lhs: a,
        rhs: AstNode(kind: IntScalar, intVal: b)
      )

# func `[]=`*[N: static int](a: var AstNode, coords: array[N, Coord], b: AstNode) =
#   assert a.kind == Output
#   a.ast_versions.add b

proc `+=`*(a: var AstNode, b: AstNode) =
  assert a.kind notin {Input, IntScalar, FloatScalar}
  if a.kind notin {Output, LVal}:
    a = AstNode(
          ctHash: genHash(),
          kind: LVal,
          symLVal: "localvar__" & $a.ctHash, # Generate unique symbol
          version: 1,
          prev_version: AstNode(
            cthash: a.ctHash,
            kind: Assign,
            lhs: AstNode(
              ctHash: a.ctHash, # Keep the hash
              kind: LVal,
              symLVal: "localvar__" & $a.ctHash, # Generate unique symbol
              version: 0,
              prev_version: nil,
            ),
            rhs: a
          )
    )
  if a.kind == Output:
    a = AstNode(
      ctHash: genHash(),
      kind: Output,
      symLVal: a.symLVal, # Keep original unique symbol
      version: a.version + 1,
      prev_version: AstNode(
        ctHash: a.ctHash,
        kind: Assign,
        lhs: a,
        rhs: a + b
      )
    )
  else:
    a = AstNode(
      ctHash: genHash(),
      kind: LVal,
      symLVal: a.symLVal, # Keep original unique symbol
      version: a.version + 1,
      prev_version: AstNode(
        ctHash: a.ctHash,
        kind: Assign,
        lhs: a,
        rhs: a + b
      )
    )

# ###########################
#
#     AST utilities
#
# ###########################
import strutils

proc `$`(ast: AstNode): string =
  proc inspect(ast: AstNode, indent: int): string =
    result.add '\n' & repeat(' ', indent) & $ast.kind & " (id: " & $hash(ast) & ')'
    let indent = indent + 2
    case ast.kind
    of Input:
      result.add '\n' & repeat(' ', indent) & "paramId \"" & $ast.symId & "\""
    of Output, LVal:
      result.add '\n' & repeat(' ', indent) & "symLVal \"" & ast.symLVal & "\""
      result.add '\n' & repeat(' ', indent) & "version \"" & $ast.version & "\""
      if ast.prev_version.isNil:
        result.add '\n' & repeat(' ', indent) & "prev_version: nil"
      else:
        result.add repeat(' ', indent) & inspect(ast.prev_version, indent)
    of IntScalar:
      result.add '\n' & repeat(' ', indent) & $ast.intVal
    of FloatScalar:
      result.add '\n' & repeat(' ', indent) & $ast.floatVal
    of Assign, Add, Mul:
      result.add repeat(' ', indent) & inspect(ast.lhs, indent)
      result.add repeat(' ', indent) & inspect(ast.rhs, indent)

  result = inspect(ast, 0)

proc replaceType*(ast: NimNode, to_replace: NimNode, replacements: NimNode): NimNode =
  # Args:
  #   - The full syntax tree
  #   - replacement type
  #   - type to replace
  proc inspect(node: NimNode): NimNode =
    case node.kind:
    of {nnkIdent, nnkSym}: return node
    of nnkEmpty: return node
    of nnkLiterals: return node
    of nnkIdentDefs:
      let i = node.len - 2 # Type position
      if node[i] == to_replace:
        result = node.copyNimTree()
        result[i] = replacements
        return
      else:
        return node
    else:
      var rTree = node.kind.newTree()
      for child in node:
        rTree.add inspect(child)
      return rTree
  result = inspect(ast)

# ###########################
#
#     Compile AST to Nim
#
# ###########################

proc walkASTGeneric(
    ast: AstNode,
    params: seq[NimNode],
    visited: var Table[AstNode, NimNode],
    stmts: var NimNode): NimNode =
  ## Recursively walk the AST
  ## Append the corresponding Nim AST for generic instructions
  ## and returns a LVal, Output or expression
  case ast.kind:
    of Input:
      return params[ast.symId]
    of IntScalar:
      return newLit(ast.intVal)
    of FloatScalar:
      return newLit(ast.floatVal)
    of Output, LVal:
      let sym = newIdentNode(ast.symLVal)
      if ast in visited:
        return sym
      elif ast.prev_version.isNil:
        visited[ast] = sym
        return sym
      else:
        visited[ast] = sym
        var blck = newStmtList()
        let expression = walkASTGeneric(ast.prev_version, params, visited, blck)
        stmts.add blck
        if not(expression.kind == nnkIdent and eqIdent(sym, expression)):
          stmts.add newAssignment(
            newIdentNode(ast.symLVal),
            expression
          )
        return newIdentNode(ast.symLVal)
    of Assign:
      if ast in visited:
        return visited[ast]

      # Workaround compileTime table not finding keys
      # https://github.com/mratsim/compute-graph-optim/issues/1
      for key in visited.keys():
        if hash(key) == hash(ast):
          return visited[key]

      echo "Hash/Type: ", ast.hash, "/", ast.kind, " visited: ", ast in visited
      var varAssign = false

      if ast.lhs notin visited and
            ast.lhs.kind == LVal and
            ast.lhs.prev_version.isNil and
            ast.rhs notin visited:
          varAssign = true

      var rhsStmt = newStmtList()
      let rhs = walkASTGeneric(ast.rhs, params, visited, rhsStmt)
      stmts.add rhsStmt

      var lhsStmt = newStmtList()
      let lhs = walkASTGeneric(ast.lhs, params, visited, lhsStmt)
      stmts.add lhsStmt

      lhs.expectKind(nnkIdent)
      if varAssign:
        stmts.add newVarStmt(lhs, rhs)
      else:
        stmts.add newAssignment(lhs, rhs)
      # visited[ast] = lhs # Already done
      return lhs

    of Add, Mul:
      if ast in visited:
        return visited[ast]

      # Workaround compileTime table not finding keys
      # https://github.com/mratsim/compute-graph-optim/issues/1
      for key in visited.keys():
        if hash(key) == hash(ast):
          return visited[key]

      echo "Hash/Type: ", ast.hash, "/", ast.kind, " visited: ", ast in visited

      var callStmt = nnkCall.newTree()
      var lhsStmt = newStmtList()
      var rhsStmt = newStmtList()

      let lhs = walkASTGeneric(ast.lhs, params, visited, lhsStmt)
      let rhs = walkASTGeneric(ast.rhs, params, visited, rhsStmt)

      stmts.add lhsStmt
      stmts.add rhsStmt

      case ast.kind
      of Add: callStmt.add newidentNode"+"
      of Mul: callStmt.add newidentNode"*"
      else: raise newException(ValueError, "Unreachable code")

      callStmt.add lhs
      callStmt.add rhs

      let memloc = genSym(nskLet, "memloc_")
      stmts.add newLetStmt(memloc, callStmt)
      visited[ast] = memloc
      return memloc

proc walkASTSimd(
    ast: AstNode,
    arch: SimdArch,
    params: seq[NimNode],
    visited: var Table[AstNode, NimNode],
    stmts: var NimNode): NimNode =
  ## Recursively walk the AST
  ## Append the corresponding Nim AST for SIMD specialized instructions
  ## and returns a LVal, Output or expression
  case ast.kind:
    of Input:
      return params[ast.symId]
    of IntScalar:
      return newCall(SimdTable[arch][simdBroadcast], newLit(ast.intVal))
    of FloatScalar:
      return newCall(SimdTable[arch][simdBroadcast], newLit(ast.floatVal))
    of Output, LVal:
      let sym = newIdentNode(ast.symLVal)
      if ast in visited:
        return sym
      elif ast.prev_version.isNil:
        visited[ast] = sym
        return sym
      else:
        visited[ast] = sym
        var blck = newStmtList()
        let expression = walkASTSimd(ast.prev_version, arch, params, visited, blck)
        stmts.add blck
        if not(expression.kind == nnkIdent and eqIdent(sym, expression)):
          stmts.add newAssignment(
            newIdentNode(ast.symLVal),
            expression
          )
        return newIdentNode(ast.symLVal)
    of Assign:
      if ast in visited:
        return visited[ast]

      # Workaround compileTime table not finding keys
      # https://github.com/mratsim/compute-graph-optim/issues/1
      for key in visited.keys():
        if hash(key) == hash(ast):
          return visited[key]

      echo "Hash/Type: ", ast.hash, "/", ast.kind, " visited: ", ast in visited
      var varAssign = false

      if ast.lhs notin visited and
            ast.lhs.kind == LVal and
            ast.lhs.prev_version.isNil and
            ast.rhs notin visited:
          varAssign = true

      var rhsStmt = newStmtList()
      let rhs = walkASTSimd(ast.rhs, arch, params, visited, rhsStmt)
      stmts.add rhsStmt

      var lhsStmt = newStmtList()
      let lhs = walkASTSimd(ast.lhs, arch, params, visited, lhsStmt)
      stmts.add lhsStmt

      lhs.expectKind(nnkIdent)
      if varAssign:
        stmts.add newVarStmt(lhs, rhs)
      else:
        stmts.add newAssignment(lhs, rhs)
      # visited[ast] = lhs # Already done
      return lhs

    of Add, Mul:
      if ast in visited:
        return visited[ast]

      # Workaround compileTime table not finding keys
      # https://github.com/mratsim/compute-graph-optim/issues/1
      for key in visited.keys():
        if hash(key) == hash(ast):
          return visited[key]

      echo "Hash/Type: ", ast.hash, "/", ast.kind, " visited: ", ast in visited

      var callStmt = nnkCall.newTree()
      var lhsStmt = newStmtList()
      var rhsStmt = newStmtList()

      let lhs = walkASTSimd(ast.lhs, arch, params, visited, lhsStmt)
      let rhs = walkASTSimd(ast.rhs, arch, params, visited, rhsStmt)

      stmts.add lhsStmt
      stmts.add rhsStmt

      case ast.kind
      of Add: callStmt.add SimdTable[arch][simdAdd]
      of Mul: callStmt.add SimdTable[arch][simdMul]
      else: raise newException(ValueError, "Unreachable code")

      callStmt.add lhs
      callStmt.add rhs

      let memloc = genSym(nskLet, "memloc_")
      stmts.add newLetStmt(memloc, callStmt)
      visited[ast] = memloc
      return memloc

proc initParams(
       procDef,
       resultType: NimNode
       ): tuple[
            ids: seq[NimNode],
            ptrs, simds: tuple[inParams, outParams: seq[NimNode]],
            length: NimNode,
            initStmt: NimNode
          ] =
  # Get the idents from proc definition. We order the same as proc def
  # Start with non-result
  # We work at simd vector level
  result.initStmt = newStmtList()
  let type0 = newCall(
    newIdentNode"type",
    nnkBracketExpr.newTree(
      procDef[0][3][1][0],
      newLit 0
    )
  )

  for i in 1 ..< procDef[0][3].len: # Proc formal params
    let iddefs = procDef[0][3][i]
    for j in 0 ..< iddefs.len - 2:
      let ident = iddefs[j]
      result.ids.add ident
      let raw_ptr = newIdentNode($ident & "_raw_ptr")
      result.ptrs.inParams.add raw_ptr

      if j == 0:
        result.length = quote do: `ident`.len
      else:
        let len0 = result.length
        result.initStmt.add quote do:
          assert `len0` == `ident`.len
      result.initStmt.add quote do:
        let `raw_ptr` = cast[ptr UncheckedArray[`type0`]](`ident`[0].unsafeAddr)
      result.simds.inParams.add newIdentNode($ident & "_simd")

  # Now add the result idents
  # We work at simd vector level
  let len0 = result.length

  if resultType.kind == nnkEmpty:
    discard
  elif resultType.kind == nnkTupleTy:
    for i in 0 ..< resultType.len:
      let iddefs = resultType[i]
      for j in 0 ..< iddefs.len - 2:
        let ident = iddefs[j]
        result.ids.add ident
        let raw_ptr = newIdentNode($ident & "_raw_ptr")
        result.ptrs.outParams.add raw_ptr

        let res = nnkDotExpr.newTree(
                    newIdentNode"result",
                    iddefs[j]
                  )
        result.initStmt.add quote do:
          `res` = newSeq[`type0`](`len0`)
          let `raw_ptr` = cast[ptr UncheckedArray[`type0`]](`res`[0].unsafeAddr)

        result.simds.outParams.add newIdentNode($ident & "_simd")

proc vectorize(
      funcName: NimNode,
      ptrs, simds: tuple[inParams, outParams: seq[NimNode]],
      len: NimNode,
      arch: SimdArch, alignNeeded, unroll_factor: int): NimNode =
  ## Vectorizing macro
  ## Apply a SIMD function on all elements of an array
  ## This deals with:
  ##   - indexing
  ##   - unrolling
  ##   - alignment
  ##   - any number of parameters and result

  # It does the same as the following templates
  #
  # template vectorize(
  #       wrapped_func,
  #       funcname: untyped,
  #       arch: static SimdArch,
  #       alignNeeded,
  #       unroll_factor: static int) =
  #   proc funcname(dst, src: ptr UncheckedArray[float32], len: Natural) =
  #
  #     template srcAlign {.dirty.} = cast[ByteAddress](src[idx].addr) and (alignNeeded - 1)
  #     template dstAlign {.dirty.} = cast[ByteAddress](dst[idx].addr) and (alignNeeded - 1)
  #
  #     doAssert srcAlign == dstAlign
  #
  #     # Loop peeling, while not aligned to required alignment
  #     var idx = 0
  #     while srcAlign() != 0:
  #       dst[idx] = wrapped_func(src[idx])
  #       inc idx
  #
  #     # Aligned part
  #     {.pragma: restrict, codegenDecl: "$# __restrict__ $#".}
  #     let srca {.restrict.} = assume_aligned cast[ptr UncheckedArray[float32]](src[idx].addr)
  #     let dsta {.restrict.} = assume_aligned cast[ptr UncheckedArray[float32]](dst[idx].addr)
  #
  #     let newLen = len - idx
  #     let unroll_stop = newLen.round_down_power_of_2(unroll_factor)
  #     for i in countup(0, unroll_stop - 1, unroll_factor):
  #       simd(
  #         arch, simdStoreA,
  #         dst[i].addr,
  #         wrapped_func(
  #           simd(arch, simdLoadA, src[i].addr)
  #         )
  #       )
  #
  #     # Unrolling remainder
  #     for i in unroll_stop ..< len:
  #       dst[i] = wrapped_func(src[i])

  template alignmentOffset(p: NimNode, idx: NimNode): untyped {.dirty.}=
    quote:
      cast[ByteAddress](`p`[`idx`].addr) and (`alignNeeded` - 1)

  result = newStmtList()

  block: # Alignment
    let align0 = alignmentOffset(ptrs.inParams[0], newLit 0)
    for i in 1 ..< ptrs.inParams.len:
      let align_i = alignmentOffset(ptrs.inParams[i], newLit 0)
      result.add quote do:
        doAssert `align0` == `align_i`
    for outparam in ptrs.outParams:
      let align_i =  alignmentOffset(outparam, newLit 0)
      result.add quote do:
        doAssert `align0` == `align_i`

  let idxPeeling = newIdentNode("idxPeeling_")


  proc elems(idx: NimNode, simd: bool): tuple[fcall, dst, dst_init, dst_assign: NimNode] =
    ## Note we need a separate ident node for each for loops
    ## otherwise the C codegen is wrong
    # Src params / Function call
    result.fcall = nnkCall.newTree()
    result.fcall.add funcName
    for p in ptrs.inParams:
      let elem = nnkBracketExpr.newTree(p, idx)
      if not simd:
        result.fcall.add elem
      else:
        result.fcall.add newCall(
          SimdTable[arch][simdLoadU], # Hack: should be aligned but no control over alignment in seq[T]
          newCall(
            newidentNode"addr",
            elem
          )
        )

    # Destination params
    # Assuming we have a function called the following way
    # (r0, r1) = foo(s0, s1)
    # We can use tuples for the non-SIMD part
    # but we will need temporaries for the SIMD part
    # before calling simdStore

    # temp variable around bug? can't use result.dst_init[0].add, type mismatch on tuple sig
    var dst_init = nnkVarSection.newTree(
      nnkIdentDefs.newTree()
    )
    result.dst_assign = newStmtList()

    if ptrs.outParams.len > 1:
      result.dst = nnkPar.newTree()
      for p in ptrs.outParams:
        let elem = nnkBracketExpr.newTree(p, idx)
        if not simd:
          result.dst.add elem
        else:
          let tmp = newIdentNode($p & "_simd")
          result.dst.add tmp
          dst_init[0].add nnkPragmaExpr.newTree(
            tmp,
            nnkPragma.newTree(
              newIdentNode"noInit"
              )
          )
          result.dst_assign.add newCall(
            SimdTable[arch][simdStoreU], # Hack: should be aligned but no control over alignment in seq[T]
            newCall(
              newidentNode"addr",
              elem
            ),
            tmp
          )
    elif ptrs.outParams.len == 1:
      let elem = nnkBracketExpr.newTree(ptrs.outParams[0], idx)
      if not simd:
        result.dst = elem
      else:
        let tmp = newIdentNode($ptrs.outParams[0] & "_simd")
        result.dst = tmp
        result.dst_assign.add newCall(
          SimdTable[arch][simdStoreU], # Hack: should be aligned but no control over alignment in seq[T]
          elem,
          tmp
        )

    dst_init[0].add SimdTable[arch][simdType]
    dst_init[0].add newEmptyNode()

    result.dst_init = dst_init

  block: # Loop peeling
    let idx = newIdentNode("idx_")
    result.add newVarStmt(idxPeeling, newLit 0)
    let whileTest = nnkInfix.newTree(
      newIdentNode"!=",
      alignmentOffset(ptrs.inParams[0], idxPeeling),
      newLit 0
    )
    var whileBody = newStmtList()
    let (fcall, dst, _, _) = elems(idx, simd = false)

    whileBody.add newLetStmt(idx, idxPeeling)
    if ptrs.outParams.len > 0:
      whileBody.add newAssignment(dst, fcall)
    else:
      whileBody.add fcall
    whileBody.add newCall(newIdentNode"inc", idxPeeling)

    result.add nnkWhileStmt.newTree(
      whileTest,
      whileBody
    )

  let unroll_stop = newIdentNode("unroll_stop_")
  block: # Aligned part
    let idx = newIdentNode("idx_")
    result.add quote do:
      let `unroll_stop` = round_down_power_of_2(
        `len` - `idxPeeling`, `unroll_factor`)

    let (fcall, dst, dst_init, dst_assign) = elems(idx, simd = true)
    if ptrs.outParams.len > 0:
      result.add dst_init

    var forStmt = nnkForStmt.newTree()
    forStmt.add idx
    forStmt.add newCall(
      newIdentNode"countup",
      idxPeeling,
      nnkInfix.newTree(
        newIdentNode"-",
        unroll_stop,
        newLit 1
      ),
      newLit unroll_factor
    )
    if ptrs.outParams.len > 0:
      forStmt.add nnkStmtList.newTree(
        newAssignment(dst, fcall),
        dst_assign
      )
    else:
      forStmt.add fcall
    result.add forStmt

  block: # Remainder
    let idx = newIdentNode("idx_")
    var forStmt = nnkForStmt.newTree()
    forStmt.add idx
    forStmt.add nnkInfix.newTree(
        newIdentNode"..<",
        unroll_stop,
        len
      )
    let (fcall, dst, _, _) = elems(idx, simd = false)
    if ptrs.outParams.len > 0:
      forStmt.add newAssignment(dst, fcall)
    else:
      forStmt.add fcall
    result.add forStmt

  # echo result.toStrLit

proc innerProcGen(
    genSimd: bool, arch: SimdArch,
    io: varargs[AstNode],
    ids: seq[NimNode],
    resultType: NimNode,
    ): NimNode =
  # Does topological ordering and dead-code elimination
  result = newStmtList()
  var visitedNodes = initTable[AstNode, NimNode]()

  for i, inOutVar in io:
    if inOutVar.kind != Input:
      if inOutVar.kind in {Output, LVal}:
        let sym = block:
          if genSimd:
            walkASTSimd(inOutVar, arch, ids, visitedNodes, result)
          else:
            walkASTGeneric(inOutVar, ids, visitedNodes, result)
        sym.expectKind nnkIdent
        if resultType.kind == nnkTupleTy:
          result.add newAssignment(
            nnkDotExpr.newTree(
              newIdentNode"result",
              ids[i]
            ),
            sym
          )
        else:
          result.add newAssignment(
            newIdentNode"result",
            sym
          )
      else:
        let expression =  block:
          if genSimd:
            walkASTSimd(inOutVar, arch, ids, visitedNodes, result)
          else:
            walkASTGeneric(inOutVar, ids, visitedNodes, result)
        if resultType.kind == nnkTupleTy:
          result.add newAssignment(
            nnkDotExpr.newTree(
              newIdentNode"result",
              ids[i]
            ),
            expression
          )
        else:
          result.add newAssignment(
            newIdentNode"result",
            expression
          )

macro compile(arch: static SimdArch, io: static varargs[AstNode], procDef: untyped): untyped =
  # Note: io must be an array - https://github.com/nim-lang/Nim/issues/10691

  # compile([a, b, c, bar, baz, buzz]):
  #   proc foobar[T](a, b, c: T): tuple[bar, baz, buzz: T]
  #
  # StmtList
  #   ProcDef
  #     Ident "foobar"
  #     Empty
  #     GenericParams
  #       IdentDefs
  #         Ident "T"
  #         Empty
  #         Empty
  #     FormalParams
  #       TupleTy
  #         IdentDefs
  #           Ident "bar"
  #           Ident "baz"
  #           Ident "buzz"
  #           Ident "T"
  #           Empty
  #       IdentDefs
  #         Ident "a"
  #         Ident "b"
  #         Ident "c"
  #         Ident "T"
  #         Empty
  #     Empty
  #     Empty
  #     Empty

  # echo procDef.treerepr

  ## Sanity checks
  procDef.expectkind(nnkStmtList)
  assert procDef.len == 1, "Only 1 statement is allowed, the function definition"
  procDef[0].expectkind({nnkProcDef, nnkFuncDef})
  # TODO: check that the function inputs are in a symbol table?
  procDef[0][6].expectKind(nnkEmpty)

  let resultTy = procDef[0][3][0]
  let (ids, ptrs, simds, length, initParams) = initParams(procDef, resultTy)

  # echo initParams.toStrLit()

  let seqT = nnkBracketExpr.newTree(
    newIdentNode"seq", newIdentNode"float32"
  )

  # We create the inner SIMD proc, specialized to a SIMD architecture
  # In the inner proc we shadow the original idents ids.
  let simdBody = innerProcGen(
    genSimd = true,
    arch = arch,
    io = io,
    ids = ids,
    resultType = resultTy
  )

  var simdProc =  if arch == Sse:
                    procDef[0].replaceType(seqT, newIdentNode"m128")
                  else:
                    procDef[0].replaceType(seqT, newIdentNode"m256")


  simdProc[6] = simdBody   # Assign to proc body
  # echo simdProc.toStrLit

  # We create the inner generic proc
  let genericBody = innerProcGen(
    genSimd = false,
    arch = arch,
    io = io,
    ids = ids,
    resultType = resultTy
  )

  var genericProc = procDef[0].replaceType(seqT, newIdentNode"float32")
  genericProc[6] = genericBody   # Assign to proc body
  echo genericProc.toStrLit

  # We vectorize the inner proc to apply to an contiguous array
  var vecBody: NimNode
  if arch == Sse:
    vecBody = vectorize(
        procDef[0][0],
        ptrs, simds,
        length,
        arch, 4, 4    # We require 4 alignment as a hack to keep seq[T] and use unaligned load/store in code
      )
  else:
    vecBody = vectorize(
        procDef[0][0],
        ptrs, simds,
        length,
        arch, 4, 8    # We require 4 alignment as a hack to keep seq[T] and use unaligned load/store in code
      )

  result = procDef.copyNimTree()
  let resBody = newStmtList()
  resBody.add initParams
  resBody.add genericProc
  resBody.add simdProc
  resBody.add vecBody
  result[0][6] = resBody

  # echo result.toStrLit

# ###########################
#
# Syntax revamp macro/pragma
#
# ###########################

proc matchAST(overload, signature: NimNode): bool =
  proc inspect(overload, signature: NimNode, match: var bool) =
    # echo "overload: ", overload.kind, " - match status: ", match
    if overload.kind in {nnkIdent, nnkSym} and overload.eqident("AstNode"):
      # AstNode match with any type
      # It should especially match with seq[T] which is of kind nnkBracketExpr
      return

    # Return early when not matching
    if overload.kind != signature.kind:
      match = false
    if overload.len != signature.len:
      match = false
    if match == false:
      return

    case overload.kind:
    of {nnkIdent, nnkSym}:
      match = eqIdent(overload, signature)
    of nnkEmpty:
      discard
    else:
      for i in 0 ..< overload.len:
        inspect(overload[i], signature[i], match)

  result = true
  inspect(overload, signature, result)

proc resolveASToverload(overloads, formalParams: NimNode): NimNode =
  if overloads.kind == nnkSym:
    result = overloads.getImpl()
    result[3].expectKind nnkFormalParams
    return
  else:
    overloads.expectKind(nnkClosedSymChoice)
    for o in overloads:
      let implSig = o.getImpl[3]
      implSig.expectKind nnkFormalParams
      let match = implSig.matchAST(formalParams)
      if match:
        return o
    raise newException(ValueError, "no matching overload found")

macro generate(ast_routine: typed, signature: untyped): untyped =
  let formalParams = signature[0][3]
  let ast = ast_routine.resolveASToverload(formalParams)

  # Get the routine signature
  let sig = ast.getImpl[3]
  sig.expectKind(nnkFormalParams)

  # Get all inputs
  var inputs: seq[NimNode]
  for idx_identdef in 1 ..< sig.len:
    let identdef = sig[idx_identdef]
    doAssert identdef[^2].eqIdent"AstNode"
    identdef[^1].expectKind(nnkEmpty)
    for idx_ident in 0 .. identdef.len-3:
      inputs.add genSym(nskLet, $identdef[idx_ident] & "_")

  # Allocate inputs
  result = newStmtList()
  proc ct(ident: NimNode): NimNode =
    nnkPragmaExpr.newTree(
      ident,
      nnkPragma.newTree(
        ident"compileTime"
      )
    )

  for i, in_ident in inputs:
    result.add newLetStmt(
      ct(in_ident),
      newCall("input", newLit i)
    )

  # Call the AST routine
  let call = newCall(ast, inputs)
  var callAssign: NimNode
  case sig[0].kind
  of nnkEmpty: # Case 1: no result
    result.add call
  # Compile-time tuple destructuring is bugged - https://github.com/nim-lang/Nim/issues/11634
  # of nnkTupleTy: # Case 2: tuple result
  #   callAssign = nnkVarTuple.newTree()
  #   for identdef in sig[0]:
  #     doAssert identdef[^2].eqIdent"AstNode"
  #     identdef[^1].expectKind(nnkEmpty)
  #     for idx_ident in 0 .. identdef.len-3:
  #       callAssign.add ct(identdef[idx_ident])
  #   callAssign.add newEmptyNode()
  #   callAssign.add call
  #   result.add nnkLetSection.newTree(
  #     callAssign
  #   )
  else: # Case 3: single return value
    callAssign = ct(genSym(nskLet, "callResult_"))
    result.add newLetStmt(
      callAssign, call
    )

  # Collect all the input/output idents
  var io = inputs
  case sig[0].kind
  of nnkEmpty:
    discard
  of nnkTupleTy:
    var idx = 0
    for identdef in sig[0]:
      for idx_ident in 0 .. identdef.len-3:
        io.add nnkBracketExpr.newTree(
          callAssign[0],
          newLit idx
        )
        inc idx
  else:
    io.add callAssign

  result.add quote do:
    compile(Sse, `io`, `signature`)

  echo result.toStrlit
# ###########################
#
#         Codegen
#
# ###########################

proc foobar(a: AstNode, b, c: AstNode): tuple[bar: AstNode, baz, buzz: AstNode] =

  let foo = a + b + c

  # Don't use in-place updates
  # https://github.com/nim-lang/Nim/issues/11637
  let bar = foo * 2

  var baz = foo * 3
  var buzz = baz

  buzz += a * 1000
  baz += b
  buzz += b

  result.bar = bar
  result.baz = baz
  result.buzz = buzz

proc foobar(a: int, b, c: int): tuple[bar, baz, buzz: int] =
  echo "Overloaded proc to test bindings"
  discard

generate foobar:
  proc foobar(a: seq[float32], b, c: seq[float32]): tuple[bar: seq[float32], baz, buzz: seq[float32]]

# Note to use aligned store, SSE requires 16-byte alignment and AVX 32-byte alignment
# Unfortunately there is no way with normal seq to specify that (pending destructors)
# As a hack, we use the unaligned load and store simd, and a required alignment of 4,
# in practice we define our own tensor type
# with aligned allocator

import sequtils

let
  len = 10
  u = newSeqWith(len, 1'f32)
  v = newSeqWith(len, 2'f32)
  w = newSeqWith(len, 3'f32)

let (pim, pam, poum) = foobar(u, v, w)

echo pim  # 12
echo pam  # 20
echo poum # 10020

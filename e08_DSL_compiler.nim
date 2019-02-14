# First try at an expressive DSL
#
# At a high level the DSL is a collection of functions to be applied to tensors (shallow embedding).
#
# Each is translated into elemental operations like:
#   - Addition, multiplication, ...
#   - sin, cos, exp, log, ...
# to build an AST tree (deep embedding)
#
# We assume that those elemental operations can be composed efficiently
# and that we do not need to implement a coarse-grained matrix multiplication
# or convolution to reach state-of-the art performance (i.e. OpenBLAS/MKL/BLIS and MKL-DNN).
#
# I.e. the DSL is functional, extensible and composable
#      and the AST classically uses ADTs/object variants.
#      This combines shallow and deep embedding.
#      It's actually equivalent to a tagless final approach
#      with only a single interpreter: Expression --> AST node
#
# As we want to be able to build an AST at compile-time (AOT) and runtime (JIT and dynamic graphs)
# We can't use generics (for generic over T at runtime and NimNode at compile-time),
# and methods don't work at compile-time.
#
# Library users can extend by implementing new functions. AST should be expressive enough
# to be used for any tensor functions and also optionally include debugging fields.

import macros

type
  AstNodeKind = enum
    # Elemental Op Kind
    Sym         # Symbol, mark an input tensor node
    IntScalar   # Mark an integer scalar that will be broadcasted
    FloatScalar # Mark a float scalar that will be broadcasted
    Add         # Elementwise add
    Mul         # Elementwise mul

  AstNode = ref object
    case kind: AstNodeKind
    of Sym:
      symbol: string   # Avoid alloc? single char? NimNode? static?
    of IntScalar:
      intVal: int      # type to use? int64?
    of FloatScalar:
      floatVal: float
    of Add, Mul:
      lhs, rhs: AstNode

  Expr = object
    # Expression compiler
    ast: AstNode

func sym*(symbol: string): Expr =
  Expr(ast: AstNode(kind: Sym, symbol: symbol))

func `+`*(a, b: Expr): Expr =
  Expr(ast: AstNode(kind: Add, lhs: a.ast, rhs: b.ast))

func `*`*(a, b: Expr): Expr =
  Expr(ast: AstNode(kind: Mul, lhs: a.ast, rhs: b.ast))

func `*`*(a: Expr, b: SomeInteger): Expr =
  Expr(
    ast: AstNode(
      kind: Mul,
      lhs: a.ast,
      rhs: AstNode(kind: IntScalar, intVal: b)
    )
  )

static: # Check if working at compile-time
  let foo = sym("a") + sym("b") + sym("c")
  let bar = foo * 2
  echo bar.repr

block: # Check if working at runtime
  let foo = sym("a") + sym("b") + sym("c")
  let bar = foo * 2
  echo bar.repr

# ###########################
#
#     Compile AST to Nim
#
# ###########################

import macros, tables, sequtils

proc walkAST(e: AstNode): NimNode =
  ## Recursively walk the expression AST
  ## Append the corresponding Nim AST
  ## Returns the new expression and node to reiterate from

  case e.kind:
  of Sym:
    return newIdentNode e.symbol
  of IntScalar:
    return newLit e.intVal
  of FloatScalar:
    return newLit e.floatVal
  of Add:
    var callTree = nnkCall.newTree()
    callTree.add newIdentNode"+"
    callTree.add e.lhs.walkAST
    callTree.add e.rhs.walkAST
    return callTree
  of Mul:
    var callTree = nnkCall.newTree()
    callTree.add newIdentNode"*"
    callTree.add e.lhs.walkAST
    callTree.add e.rhs.walkAST
    return callTree

macro compile(expression: static Expr, func_def: untyped): untyped =
  # bar.compile:
  #   func foobar[T](a, b, c: T): T
  #
  # func_def has the following form
  #
  #   StmtList
  #     FuncDef
  #       Ident "foobar"
  #       Empty
  #       GenericParams
  #         IdentDefs
  #           Ident "T"
  #           Empty
  #           Empty
  #       FormalParams
  #         Ident "T"
  #         IdentDefs
  #           Ident "a"
  #           Ident "b"
  #           Ident "c"
  #           Ident "T"
  #           Empty
  #       Empty
  #       Empty
  #       Empty <---- proc body

  ## Sanity checks
  func_def.expectkind(nnkStmtList)
  assert func_def.len == 1, "Only 1 statement is allowed, the function definition"
  func_def[0].expectkind({nnkProcDef, nnkFuncDef})
  # TODO: check that the function inputs are in a symbol table?
  func_def[0][6].expectKind(nnkEmpty)

  result = func_def.copyNimTree()         # Copy function definition
  result[0][6] = expression.ast.walkAST   # Assign body

  echo result.toStrLit

let foo{.compileTime.} = sym("a") + sym("b") + sym("c")
let bar{.compileTime.} = foo * 2
bar.compile:
  func foobar[T](a, b, c: T): T

echo foobar(1, 2, 3)
echo (1 + 2 + 3) * 2
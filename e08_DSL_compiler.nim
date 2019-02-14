# First try at an expressive DSL
#
# At a high level the DSL is a collection of functions.
#
# Each is translated into elemental operations like:
#   - Addition, multiplication, ...
#   - sin, cos, exp, log, ...
#
# We assume that those elemental operations can be composed efficiently
# and that we do not need to implement a coarse-grained matrix multiplication
# or convolution to reach state-of-the art performance (i.e. OpenBLAS/MKL/BLIS and MKL-DNN).
#
# I.e. the DSL is functional and the AST classically uses ADTs/object variants.
#      It's actually equivalent to a tagless final approach
#      with onyl a single interpreter: Expression --> AST node
#
# As we want to be able to build an AST at compile-time (AOT) and runtime (JIT and dynamic graphs)
# We can't use generics (for generic over T at runtime and NimNode at compile-time),
# and methods don't work at compile-time.
#
# Library users can extend by implementing new functions. AST should be expressive enough
# to be used for any tensor functions and also optionally include debugging fields.

import macros

type
  IRNodeKind = enum
    # Elemental Op Kind
    Sym         # Symbol, mark an input tensor node
    IntScalar   # Mark an integer scalar that will be broadcasted
    FloatScalar # Mark a float scalar that will be broadcasted
    Add         # Elementwise add
    Mul         # Elementwise mul

  IRNode = ref object
    case kind: IRNodeKind
    of Sym:
      symbol: string   # Avoid alloc? single char? NimNode? static?
    of IntScalar:
      intVal: int      # type to use? int64?
    of FloatScalar:
      floatVal: float
    of Add, Mul:
      lhs, rhs: IRNode

  Expr = object
    # Expression compiler
    ast: IRNode

func sym*(symbol: string): Expr =
  Expr(ast: IRNode(kind: Sym, symbol: symbol))

func `+`*(a, b: Expr): Expr =
  Expr(ast: IRNode(kind: Add, lhs: a.ast, rhs: b.ast))

func `*`*(a, b: Expr): Expr =
  Expr(ast: IRNode(kind: Mul, lhs: a.ast, rhs: b.ast))

func `*`*(a: Expr, b: SomeInteger): Expr =
  Expr(
    ast: IRNode(
      kind: Mul,
      lhs: a.ast,
      rhs: IRNode(kind: IntScalar, intVal: b)
    )
  )

let foo = sym("a") + sym("b") + sym("c")

let bar = foo * 2

echo bar.repr
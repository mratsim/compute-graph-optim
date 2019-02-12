# Using extensible metaprogrammed ADTs

import macros, tables, hashes

proc hash*(x: NimNode): Hash =
  assert x.kind == nnkIdent
  result = hash($x)

var ExprNodeKind* {.compileTime.} = {
  # TypeTag: Arity
  newIdentNode"Lit": 0, # Terminal node, holds T
  newIdentNode"Add": 2  # Non-terminal node, holds 2 expr
}.toTable

macro register_op*(name: untyped, arity: static int): untyped =
  ExprNodeKind[name] = arity

# ######################
# Interpreters
# Evaluation interpreter

template evalLit*(node: untyped): untyped =
  assert node.kind == Lit
  node.lit

template evalAdd*(node: untyped): untyped =
  assert node.kind == Add
  node.expr2[0].eval + node.expr2[1].eval

var EvalInterpreter* {.compileTime.} = {
    # TypeTag: EvaluationRoutine
    newIdentNode"Lit": newIdentNode"evalLit",
    newIdentNode"Add": newIdentNode"evalAdd"
  }.toTable

macro register_eval*(name, eval: untyped): untyped =
  assert name in ExprNodeKind
  EvalInterpreter[name] = eval

# Pretty-print interpreter

template ppLit*(node: untyped): untyped =
  assert node.kind == Lit
  $node.lit

template ppAdd*(node: untyped): untyped =
  assert node.kind == Add
  "(" & node.expr2[0].prettyprint & " + " & node.expr2[1].prettyprint & ")"

var PPInterpreter* {.compileTime.} = {
    # TypeTag: EvaluationRoutine
    newIdentNode"Lit": newIdentNode"ppLit",
    newIdentNode"Add": newIdentNode"ppAdd"
  }.toTable

macro register_pp*(name, pp: untyped): untyped =
  assert name in ExprNodeKind
  PPInterpreter[name] = pp
# Using extensible metaprogrammed ADTs

import
  macros,
  ./e07_part1_var_ADTs_tables,
  ./e07_part2_var_ADTs_grammar

template evalMul*(node: untyped): untyped =
  assert node.kind == Mul
  node.expr2[0].eval * node.expr2[1].eval

register_op(Mul, arity=2)
register_eval(Mul, evalMul)

genExprEnum()
genAlgebra()
genEvalInterpreter()

proc `*`[T](a, b: Expr[T]): Expr[T] =
  Expr[T](kind: Mul, expr2: [a, b])

# Eager evaluation

let a = lit(2)
let b = lit(5)

let c = a + b + a
echo eval(c) # 9

let d = c * c
echo eval(d) # 81

# Pretty print

template ppMul*(node: untyped): untyped =
  assert node.kind == Mul
  "(" & node.expr2[0].prettyprint & " * " & node.expr2[1].prettyprint & ")"

register_pp(Mul, ppMul)
genPPInterpreter()

echo prettyprint(c) # ((2 + 5) + 2)
echo prettyprint(d) # (((2 + 5) + 2) * ((2 + 5) + 2))
# Object Algebra DSL with AST nodes

import macros

type
  Expr = object of RootObj

  Input[T] = object of Expr
    value: NimNode

  AddExpr[T1, T2] = object of Expr
    lhs: T1
    rhs: T2

proc input[T](value: NimNode): Input[T] =
  Input[T](value: value)

proc `+`[T1, T2](lhs: T1, rhs: T2): AddExpr[T1, T2] =
  result.lhs = lhs
  result.rhs = rhs

proc eval[T](input: Input[T]): NimNode =
  input.value

proc eval(addExpr: AddExpr): NimNode =
  newCall(ident"+", addExpr.lhs.eval, addExpr.rhs.eval)

macro foo(a, b: int): untyped =
  
  let ia = input[int](a)
  let ib = input[int](b)
  
  let c = ia + ib + ia
  echo c.repr

  let val_C = eval(c)
  echo val_C.repr

  result = val_C


let a = 1
let b = 3

let c = foo(a, b)
echo c
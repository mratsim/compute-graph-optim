type
  Expr = object of RootObj

  Input[T] = object of Expr
    value: T

  AddExpr[T1, T2] = object of Expr
    lhs: T1
    rhs: T2

proc input[T](value: T): Input[T] =
  Input[T](value: value)

proc `+`[T1, T2](lhs: T1, rhs: T2): AddExpr[T1, T2] =
  result.lhs = lhs
  result.rhs = rhs

proc `$`(input: Input): string =
  $input.value

proc `$`(addExpr: AddExpr): string =
  "(" & $addExpr.lhs & " + " & $addExpr.rhs & ")"

proc eval[T](input: Input[T]): T =
  input.value

proc eval(addExpr: AddExpr): auto =
  addExpr.lhs.eval + addExpr.rhs.eval

when true:
  let a = input(1)
  let b = input(3)

  let c = a + b + a

  echo c
  echo eval(c)
else:
  let a{.compileTime.} = input(1)
  let b{.compileTime.} = input(3)

  let c{.compileTime.} = a + b + a

  static:
    echo c
    echo eval(c)
import macros

type
  Expr = object of RootObj

  Input[T] = object of Expr
    value: T

  AddExpr[T] = object of Expr
    lhs, rhs: T

proc input[T](value: T): Input[T] =
  Input[T](value: value)

proc `+`[T](lhs, rhs: T): AddExpr[T] =
  AddExpr[T](lhs: lhs, rhs: rhs)

proc `$`(input: Input): string =
  $input.value

proc `$`(addExpr: AddExpr): string =
  "(" & $addExpr.lhs & " + " & $addExpr.rhs & ")"

proc eval[T](input: Input[T]): T =
  input.value

proc eval[T](addExpr: AddExpr[T]): auto =
  addExpr.lhs.eval + addExpr.rhs.eval

when true:
  let a = input(1)
  let b = input(3)

  let c = a + b

  echo c
  echo eval(c)
else:
  let a{.compileTime.} = input(1)
  let b{.compileTime.} = input(3)

  let c{.compileTime.} = a + b

  static:
    echo c
    echo eval(c)
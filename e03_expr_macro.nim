import macros

type
  Expr = object of RootObj

  Input = object of Expr
    value: NimNode

  AddExpr[T1, T2: Expr] = object of Expr
    lhs: T1
    rhs: T2

proc input(value: NimNode): Input =
  Input(value: value)

proc eval(input: Input): NimNode =
  input.value

proc eval(addExpr: AddExpr): NimNode =
  newCall(ident"+", addExpr.lhs.eval, addExpr.rhs.eval)

proc `+`[T1, T2: Expr](lhs: T1, rhs: T2): AddExpr[T1, T2] =
  result.lhs = lhs
  result.rhs = rhs

let a{.compileTime.} = input newLit(1) # newIdentNode"a"
let b{.compileTime.} = input newLit(3) # newIdentNode"b"

let c{.compileTime.} = a + b

let d{.compileTime.} = c + a

static:
  echo d.repr
  echo eval(d).repr

macro foo(): untyped =
  result = quote do: `d`

echo foo() # Raised upstream: https://github.com/nim-lang/Nim/issues/10626
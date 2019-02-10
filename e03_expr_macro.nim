import macros

type
  Expr = object of RootObj

  Input = object of Expr
    symbol: NimNode

  AddExpr[T1, T2: Expr] = object of Expr
    lhs: T1
    rhs: T2

proc input(symbol: NimNode): Input =
  Input(symbol: symbol)

proc eval(input: Input): NimNode =
  input.symbol

proc eval(addExpr: AddExpr): NimNode =
  newCall(ident"+", addExpr.lhs.eval, addExpr.rhs.eval)

proc `+`[T1, T2: Expr](lhs: T1, rhs: T2): AddExpr[T1, T2] =
  result.lhs = lhs
  result.rhs = rhs

let a{.compileTime.} = input newLit(1) # bindSym"x"
let b{.compileTime.} = input newLit(3) # bindSym"y"

let c{.compileTime.} = a + b

let d{.compileTime.} = c + a

static:
  echo d.repr
  echo eval(d).repr

macro foo(): untyped =
  let e = eval(d)
  result = quote do: `e`

echo foo()
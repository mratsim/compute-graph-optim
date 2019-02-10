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


let x = 12
let y = 13

# Raised: https://github.com/nim-lang/Nim/issues/10628
let a{.compileTime.} = input bindSym"x"
let b{.compileTime.} = input bindSym"y"

let c{.compileTime.} = a + b

let d{.compileTime.} = c + a

static:
  echo d.repr
  echo eval(d).repr

macro foo(): untyped =
  let e = eval(d)
  result = quote do: `e`

echo foo()
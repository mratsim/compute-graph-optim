# Typed tagless final DSL

type
  Expr[Repr] = concept x, type T
    lit(T) is Repr[T]
    `+`(Repr[T], Repr[T]) is Repr[T]

  Id[T] = object
    val: T

  Print[T] = object
    str: string

func lit[T](n: T, Repr: type[Id]): Id[T] =
  Id[T](val: n)

func `+`[T](a, b: Id[T]): Id[T] =
  Id[T](val: a.val + b.val)

func lit[T](n: T, Repr: type[Print]): Print[T] =
  Print[T](str: $n)

func `+`[T](a, b: Print[T]): Print[T] =
  Print[T](str: "(" & $a.str & " + " & $b.str & ")")

func foo(Repr: type): Repr =
  result = lit(1, Repr) + lit(2, Repr) + lit(3, Repr)

echo foo(Id).val
echo foo(Print).str

# Extend concept with "user-defined" proc.

type
  ExprUser[Repr] = concept x, type T
    x is Expr
    `*`(Repr[T], Repr[T]) is Repr[T]

func `*`[T](a, b: Id[T]): Id[T] =
  Id[T](val: a.val + b.val)

func `*`[T](a, b: Print[T]): Print[T] =
  Print[T](str: "(" & $a.str & " * " & $b.str & ")")

func bar(Repr: type): Repr =
  result = foo(Repr) * lit(10, Repr)

echo bar(Id).val
echo bar(Print).str

# Combine with other concepts

type
  Lambda[Repr] = concept x, type A, type B
    apply(Repr[A], func(a: A): B) is Repr[B]

func apply[A, B](a: Id[A], f: func(a: A): B {.nimcall.}): Id[B] =
  Id[B](val: f(a.val))

func add10[T](x: T): T = x + 10

func baz[Repr, T](e: Repr, l: func(x: T): T {.nimcall.}): Repr =
  apply(e, l)

echo baz(bar(Id), add10[int]).val
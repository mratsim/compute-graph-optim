# Typed tagless final DSL with runtime parse tree

type
  # Interpreters for the DSL
  Id[T] = object
    # Eager Evaluation
    val: T

  Print = object
    # String Evaluation
    str: string

func lit[T](n: T, Repr: type[Id]): Id[T] =
  Id[T](val: n)

func `+`[T](a, b: Id[T]): Id[T] =
  Id[T](val: a.val + b.val)

func lit[T](n: T, Repr: type[Print]): Print =
  Print(str: $n)

func `+`(a, b: Print): Print =
  Print(str: "(" & $a.str & " + " & $b.str & ")")

func foo(Repr: type): Repr =
  result = lit(1, Repr) + lit(2, Repr) + lit(3, Repr)

echo foo(Id).val
echo foo(Print).str

type
  # Parse Tree builder
  Tree = ref object of RootObj
  Leaf = ref object of Tree
    name: string
  Node[N: static int] = ref object of Tree
    name: string
    node: array[N, Tree]

template lit(n: untyped, Repr: type[Tree]): Leaf =
  Leaf(name: n.astToStr)

func `+`(a, b: Tree): Node[2] =
  new result
  result.name = "Add"
  result.node[0] = a
  result.node[1] = b

echo foo(Tree).repr
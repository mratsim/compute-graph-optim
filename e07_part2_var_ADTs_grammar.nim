# Using extensible metaprogrammed ADTs

import
  macros, tables,
  ./e07_part1_var_ADTs_tables

# dumpAstGen:
#   type
#     ExprKind* = enum
#       Lit
#       Add
    
#     Expr*[T] = ref object
#       case kind: ExprKind
#       of Lit:
#         value: T
#       of Add:
#         expr2: array[2, Expr[T]]

# Note: we group expr fields by arity as Nim doesn't allow
#       reusing the same field name in separate ``of`` branches
#       https://github.com/nim-lang/RFCs/issues/19

# dumpAstGen:
#   type
#     ExprKind* = enum
#       Lit
#       Add
#       Mul
    
#     Expr*[T] = ref object
#       case kind: ExprKind
#       of Lit:
#         value: T
#       of Add, Mul:
#         expr2: array[2, Expr[T]]

proc exported(name: string): NimNode =
  nnkPostfix.newTree(
    newIdentNode"*",
    newIdentNode name
  )

proc genericParam(symbold: string): NimNode =
  nnkGenericParams.newTree(
    newIdentDefs(
      newIdentNode"T", newEmptyNode()
    )
  )

macro genExprEnum*(): untyped =
  var enumValues = nnkEnumTy.newTree()
  enumValues.add newEmptyNode()

  var adtCases = nnkRecCase.newTree()
  # case kind: ExprKind
  adtCases.add newIdentDefs(exported"kind", ident"ExprKind")

  var arityTags = initTable[int, seq[NimNode]]()

  for kind, arity in ExprNodeKind:
    enumValues.add kind

    if arity == 0:
      # of Lit:
      #   lit: T
      adtCases.add nnkOfbranch.newTree(
        kind,
        nnkRecList.newTree(
          newIdentDefs(ident"lit", ident"T")
        )
      )
    else: # Remap table Tags -> Arity to Arity -> Tags
      arityTags.mgetOrPut(arity, @[]).add kind

  for arity, tags in arityTags:
    # of Add, Mul:
    #   expr2: array[2, Expr[T]]
    var branch = nnkOfbranch.newTree()
    branch.add tags
    branch.add nnkRecList.newTree(
      newIdentDefs(
        exported("expr" & $arity),
        nnkBracketExpr.newTree(
          ident"array", newLit arity,
          nnkBracketExpr.newTree(
            ident"Expr", ident"T"
          )
        )
      )
    )
    adtCases.add branch

  result = nnkStmtList.newTree(
    nnkTypeSection.newTree(
      # type ExprKind* = enum
      #   Lit, Add, Mul
      nnkTypeDef.newTree(
        exported "ExprKind",
        newEmptyNode(),
        enumValues
      ),
      # type Expr*[T] = ref object
      #   case kind: ExprKind
      #   of Lit:
      #     lit: T
      #   of Add, Mul:
      #     expr2: array[2, Expr[T]]
      nnkTypeDef.newTree(
        exported "Expr",
        genericParam"T",
        nnkRefTy.newTree(
          nnkObjectTy.newTree(
            newEmptyNode(),
            newEmptyNode(),
            adtCases
          )
        )
      )
    )
  )
  # echo result.toStrLit

# Algebra

template genAlgebra*(): untyped =
  proc lit*[T](x: T): Expr[T] =
    Expr[T](kind: Lit, lit: x)

  proc `+`*[T](a, b: Expr[T]): Expr[T] =
    Expr[T](kind: Add, expr2: [a, b])

# Interpreter generation


# Currently we can't pass a param "Interpreter: static Table[NimNode, NimNode]"
# to a macro due to https://github.com/nim-lang/Nim/issues/9679
# So we use a proc as workaround
proc genInterpreterImpl(funcName: NimNode, resultType: NimNode, Interpreter: Table[NimNode, NimNode]): NimNode =
  let expression = genSym(nskParam, "expression_")
  var body = nnkCaseStmt.newTree(
      nnkDotExpr.newTree(expression, ident"kind")
    )

  # Add a branch for each (expression, implementation) pair
  for kind, impl in Interpreter:
    body.add nnkOfbranch.newTree(
      kind,
      newCall(
        impl,
        expression
      )
    )
  
  result = quote do:
    proc `funcName`*[T](`expression`: Expr[T]): `resultType` =
      `body`
  echo result.toStrLit

macro genEvalInterpreter*(): untyped =
  result = genInterpreterImpl(newIdentNode"eval", newIdentNode"T", EvalInterpreter)

macro genPPInterpreter*(): untyped =
  result = genInterpreterImpl(newIdentNode"prettyprint", newIdentNode"string", PPInterpreter)
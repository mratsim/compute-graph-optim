import macros

# ###########################################
#
#         Internal Graph Representation
#
# ###########################################

# For implementation ease, controlling side-effects and parallelism
# We want a functional side-effect free language
# However in-place updates like "+=" are just too ergonomic to pass up
# we simulate those in the AST via a persistent data-structure

import hashes, random, sets
var astNodeRng {.compileTime.} = initRand(0x42)
  ## Workaround for having no UUID for AstNodes
  ## at compile-time - https://github.com/nim-lang/RFCs/issues/131

type
  AstNodeKind = enum
    ## Elemental Op Kind

    # Expressions
    IntScalar   # Mark an integer scalar that will be broadcasted
    FloatScalar # Mark a float scalar that will be broadcasted
    Add         # Elementwise add
    Mul         # Elementwise mul

    # Symbols
    Input       # Input tensor node
    Output      # Mutable output tensor node
    LVal        # Temporary allocated node
    Asgn        # Assignment statement

  AstNode = ref object
    case kind: AstNodeKind
    of Input:
      symIn: string              # Avoid alloc? single char? NimNode? static?
    of Output, LVal:
      symLval: string
      prev_version: AstNode      # Persistent data structure
    of IntScalar:
      intVal: int                # type to use? int64?
    of FloatScalar:
      floatVal: float
    of Asgn, Add, Mul:
      lhs, rhs: AstNode

    ctHash: Hash                 # Compile-Time only Hash

  Coord = object

proc genHash(): Hash =
  Hash astNodeRng.rand(high(int))

proc hash(x: AstNode): Hash {.inline.} =
  when nimvm:
    x.cthash
  else: # Take its address
    cast[Hash](x)

proc input*(symbol: string): AstNode =
  when nimvm:
    AstNode(ctHash: genHash(), kind: Input, symIn: symbol)
  else:
    AstNode(kind: Input, symIn: symbol)

proc `+`*(a, b: AstNode): AstNode =
  when nimvm:
    AstNode(ctHash: genHash(), kind: Add, lhs: a, rhs: b)
  else:
    AstNode(kind: Add, lhs: a, rhs: b)

proc `*`*(a, b: AstNode): AstNode =
  when nimvm:
    AstNode(ctHash: genHash(), kind: Mul, lhs: a, rhs: b)
  else:
    AstNode(ctHash: genHash(), kind: Mul, lhs: a, rhs: b)

proc `*`*(a: AstNode, b: SomeInteger): AstNode =
  when nimvm:
    AstNode(
        ctHash: genHash(),
        kind: Mul,
        lhs: a,
        rhs: AstNode(kind: IntScalar, intVal: b)
      )
  else:
    AstNode(
        kind: Mul,
        lhs: a,
        rhs: AstNode(kind: IntScalar, intVal: b)
      )

# func `[]=`*[N: static int](a: var AstNode, coords: array[N, Coord], b: AstNode) =
#   assert a.kind == Output
#   a.ast_versions.add b

proc `+=`*(a: var AstNode, b: AstNode) =
  assert a.kind notin {Input, IntScalar, FloatScalar}
  if a.kind notin {Output, LVal}:
    a = AstNode(
          ctHash: genHash(),
          kind: LVal,
          symLVal: "lval__" & $a.ctHash, # Generate unique symbol
          prev_version: AstNode(
            cthash: genHash(),
            kind: Asgn,
            lhs: AstNode(
              ctHash: a.ctHash, # Keep the hash
              kind: LVal,
              symLVal: "lval__" & $a.ctHash, # Generate unique symbol
              prev_version: nil,
            ),
            rhs: a
          )
    )
  if a.kind == Output:
    a = AstNode(
      ctHash: genHash(),
      kind: Output,
      symLVal: a.symLVal, # Keep original unique symbol
      prev_version: AstNode(
        ctHash: genHash(),
        kind: Asgn,
        lhs: a,
        rhs: a + b
      )
    ) 
  else:
    a = AstNode(
      ctHash: genHash(),
      kind: LVal,
      symLVal: a.symLVal, # Keep original unique symbol
      prev_version: AstNode(
        ctHash: genHash(),
        kind: Asgn,
        lhs: a,
        rhs: a + b
      )
    )

# ###########################
#
#     Print AST
#
# ###########################
import strutils

proc `$`(ast: AstNode): string =
  proc inspect(ast: AstNode, indent: int): string =
    result.add '\n' & repeat(' ', indent) & $ast.kind
    let indent = indent + 2
    case ast.kind
    of Input:
      result.add '\n' & repeat(' ', indent) & "symIn \"" & ast.symIn & "\""      
    of Output, LVal:
      result.add '\n' & repeat(' ', indent) & "symLVal \"" & ast.symLVal & "\""
      if ast.prev_version.isNil:
        result.add '\n' & repeat(' ', indent) & "prev_version: nil"
      else:
        result.add repeat(' ', indent) & inspect(ast.prev_version, indent)
    of IntScalar:
      result.add '\n' & repeat(' ', indent) & $ast.intVal
    of FloatScalar:
      result.add '\n' & repeat(' ', indent) & $ast.floatVal
    of Asgn, Add, Mul:
      result.add repeat(' ', indent) & inspect(ast.lhs, indent)
      result.add repeat(' ', indent) & inspect(ast.rhs, indent)
  result = inspect(ast, 0)

# ###########################
#
#     Compile AST to Nim
#
# ###########################

proc walkASTGeneric(e: AstNode, visited: var HashSet[AstNode]): NimNode =
  ## Recursively walk the AST
  ## Append the corresponding Nim AST for Generic CPU

  # TODO - Allocate shared expression

  case e.kind:
  of Input:
    visited.incl e
    return newIdentNode e.symIn
    # Todo if input is another function, resolve and fuse
  of Asgn:
    let symNode = newIdentNode(e.lhs.symLVal)
    if e in visited:
      return symNode

    visited.incl e
    if e.lhs in visited:
      return symNode
    elif e.lhs.prev_version.isNil:
      var asgnUpdate = newStmtList()
      if e.lhs.kind == LVal and e.lhs.prev_version.isNil:
        asgnUpdate.add newVarStmt(symNode, e.rhs.walkASTGeneric(visited))
      else:
        asgnUpdate.add newAssignment(symNode, e.rhs.walkASTGeneric(visited))
      return asgnUpdate
    else:
      var asgnUpdate = newStmtList()
      asgnUpdate.add e.lhs.prev_version.walkASTGeneric(visited)
      asgnUpdate.add newAssignment(e.lhs.walkASTGeneric(visited), e.rhs.walkASTGeneric(visited))
      return asgnUpdate
  of Output, LVal:
    let symNode = newIdentNode(e.symLVal)
    if e in visited:
      return symNode
    elif e.prev_version.isNil:
      visited.incl e
      return symNode
    else:
      visited.incl e
      return e.prev_version.walkASTGeneric(visited)
  of IntScalar:
    visited.incl e
    return newLit e.intVal
  of FloatScalar:
    visited.incl e
    return newLit e.floatVal
  of Add:
    visited.incl e
    var callTree = nnkCall.newTree()
    callTree.add newIdentNode"+"
    callTree.add e.lhs.walkASTGeneric(visited)
    callTree.add e.rhs.walkASTGeneric(visited)
    return callTree
  of Mul:
    visited.incl e
    var callTree = nnkCall.newTree()
    callTree.add newIdentNode"*"
    callTree.add e.lhs.walkASTGeneric(visited)
    callTree.add e.rhs.walkASTGeneric(visited)
    return callTree

macro compile(io: static varargs[AstNode], procDef: untyped): untyped =
  # Note: io must be an array - https://github.com/nim-lang/Nim/issues/10691

  # compile([a, b, c, bar, baz, buzz]):
  #   proc foobar[T](a, b, c: T): tuple[bar, baz, buzz: T]
  #
  # StmtList
  #   ProcDef
  #     Ident "foobar"
  #     Empty
  #     GenericParams
  #       IdentDefs
  #         Ident "T"
  #         Empty
  #         Empty
  #     FormalParams
  #       TupleTy
  #         IdentDefs
  #           Ident "bar"
  #           Ident "baz"
  #           Ident "buzz"
  #           Ident "T"
  #           Empty
  #       IdentDefs
  #         Ident "a"
  #         Ident "b"
  #         Ident "c"
  #         Ident "T"
  #         Empty
  #     Empty
  #     Empty
  #     Empty

  # echo procDef.treerepr

  ## Sanity checks
  procDef.expectkind(nnkStmtList)
  assert procDef.len == 1, "Only 1 statement is allowed, the function definition"
  procDef[0].expectkind({nnkProcDef, nnkFuncDef})
  # TODO: check that the function inputs are in a symbol table?
  procDef[0][6].expectKind(nnkEmpty)


  # Get the idents from proc definition. We order the same as proc def
  # Start with non-result
  var procIdents: seq[Nimnode]
  for i in 1 ..< procDef[0][3].len: # Proc formal params
    let iddefs = procDef[0][3][i]
    for j in 0 ..< iddefs.len - 2:
      procIdents.add iddefs[j]

  # Now add the result idents
  let resultTy = procDef[0][3][0]
  if resultTy.kind == nnkEmpty:
    discard
  elif resultTy.kind == nnkTupleTy:
    for i in 0 ..< resultTy.len:
      let iddefs = resultTy[i]
      for j in 0 ..< iddefs.len - 2:
        procIdents.add iddefs[j]
    
  # Topological ordering, dead-code elimination
  var body = newStmtList()
  var visitedNodes = initSet[AstNode]()

  for i, inOutVar in io:
    if inOutVar.kind != Input:
      if inOutVar.kind == Output:
        # Inplace mutation
        body.add nnkAsgn.newTree(
          procIdents[i],
          inOutVar.walkASTGeneric(visitedNodes)
        )
      elif inOutVar.kind == LVal:
        # It's actually a mutable output declared with
        # var foo = 1
        # foo += 1  <---- will transform into LVal

        echo inOutVar
        let varAsgnStmt = inOutVar.walkASTGeneric(visitedNodes)

        echo varAsgnStmt.treerepr 

        body.add varAsgnStmt
        body.add nnkAsgn.newTree(
          nnkDotExpr.newTree(
            newIdentNode"result",
            procIdents[i]
          ),
          varAsgnStmt[0][0][0][0]
        )
      else:
        # Expression
        if resultTy.kind == nnkTupleTy:
          body.add nnkAsgn.newTree(
            nnkDotExpr.newTree(
              newIdentNode"result",
              procIdents[i]
            ),
            inOutVar.walkASTGeneric(visitedNodes)
          )
        else:
          body.add inOutVar.walkASTGeneric(visitedNodes)

  result = procDef.copyNimTree()
  result[0][6] = body   # Assign to proc body

  echo result.toStrLit


let
  a {.compileTime.} = input("a")
  b {.compileTime.} = input("b")
  c {.compileTime.} = input("c")

  foo {.compileTime.} = a + b + c
  bar {.compileTime.} = foo * 2

var baz {.compileTime.} = foo * 3
var buzz {.compileTime.} = baz

static:
  buzz += a * 10000

compile([a, b, c, bar, baz, buzz]):
  proc foobar[T](a, b, c: T): tuple[bar, baz, buzz: T]

let (pim, pam, poum) = foobar(1, 2, 3)

echo pim # 12
echo pam # 18
echo poum # 10018
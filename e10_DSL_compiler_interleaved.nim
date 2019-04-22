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

import hashes, random, tables
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
    Assign      # Assignment statement

  AstNode = ref object
    case kind: AstNodeKind
    of Input:
      symIn: string              # Avoid alloc? single char? NimNode? static?
    of Output, LVal:
      symLval: string
      version: int
      prev_version: AstNode      # Persistent data structure
    of IntScalar:
      intVal: int                # type to use? int64?
    of FloatScalar:
      floatVal: float
    of Assign, Add, Mul:
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
          symLVal: "localvar__" & $a.ctHash, # Generate unique symbol
          version: 1,
          prev_version: AstNode(
            cthash: a.ctHash,
            kind: Assign,
            lhs: AstNode(
              ctHash: a.ctHash, # Keep the hash
              kind: LVal,
              symLVal: "localvar__" & $a.ctHash, # Generate unique symbol
              version: 0,
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
      version: a.version + 1,
      prev_version: AstNode(
        ctHash: a.ctHash,
        kind: Assign,
        lhs: a,
        rhs: a + b
      )
    ) 
  else:
    a = AstNode(
      ctHash: genHash(),
      kind: LVal,
      symLVal: a.symLVal, # Keep original unique symbol
      version: a.version + 1,
      prev_version: AstNode(
        ctHash: a.ctHash,
        kind: Assign,
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
      result.add '\n' & repeat(' ', indent) & "version \"" & $ast.version & "\""
      if ast.prev_version.isNil:
        result.add '\n' & repeat(' ', indent) & "prev_version: nil"
      else:
        result.add repeat(' ', indent) & inspect(ast.prev_version, indent)
    of IntScalar:
      result.add '\n' & repeat(' ', indent) & $ast.intVal
    of FloatScalar:
      result.add '\n' & repeat(' ', indent) & $ast.floatVal
    of Assign, Add, Mul:
      result.add repeat(' ', indent) & inspect(ast.lhs, indent)
      result.add repeat(' ', indent) & inspect(ast.rhs, indent)
  result = inspect(ast, 0)

# ###########################
#
#     Compile AST to Nim
#
# ###########################

proc walkASTGeneric(ast: AstNode, visited: var Table[AstNode, NimNode], stmts: var NimNode): NimNode =
  ## Recursively walk the AST
  ## Append the corresponding Nim AST for Generic CPU
  ## and returns a LVal, Output or expression
  case ast.kind:
    of Input:
      return newIdentNode(ast.symIn)
    of IntScalar:
      return newLit(ast.intVal)
    of FloatScalar:
      return newLit(ast.floatVal)
    of Output, LVal:
      let sym = newIdentNode(ast.symLVal)
      if ast in visited:
        return sym
      elif ast.prev_version.isNil:
        visited[ast] = sym
        return sym
      else:
        visited[ast] = sym
        var blck = newStmtList()
        let expression = walkASTGeneric(ast.prev_version, visited, blck)
        stmts.add blck
        if not(expression.kind == nnkIdent and eqIdent(sym, expression)):
          stmts.add newAssignment(
            newIdentNode(ast.symLVal),
            expression
          )
        return newIdentNode(ast.symLVal)
    of Assign, Add, Mul:
      if ast in visited:
        return # nil
      else:
        var varAssign = false

        var lhs: NimNode
        if ast.lhs notin visited:
          if ast.lhs.kind == LVal and ast.lhs.prev_version.isNil:
            varAssign = true
            visited[ast] = newIdentNode(ast.lhs.symLVal)
          var lhsStmt = newStmtList()
          lhs = walkASTGeneric(ast.lhs, visited, lhsStmt)
          stmts.add lhsStmt
        else:
          lhs = visited[ast.lhs]

        var rhs: NimNode
        if ast.rhs notin visited:
          var rhsStmt = newStmtList()
          rhs = walkASTGeneric(ast.rhs, visited, rhsStmt)
          stmts.add rhsStmt
          # visited[ast.rhs] = rhs # Done in walkAST
        else:
          rhs = visited[ast.rhs]

        if ast.kind == Assign:
          lhs.expectKind(nnkIdent)
          if varAssign:
            stmts.add newVarStmt(lhs, rhs)
          else:
            stmts.add newAssignment(lhs, rhs)
          return lhs
        else:
          var callStmt = nnkCall.newTree()
          case ast.kind
          of Add: callStmt.add newIdentNode"+"
          of Mul: callStmt.add newIdentNode"*"
          else: raise newException(ValueError, "Unreachable code")
          callStmt.add lhs
          callStmt.add rhs
          let memloc = genSym(nskLet, "memloc_")
          stmts.add newLetStmt(memloc, callStmt)
          visited[ast] = memloc
          return memloc

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
  var visitedNodes = initTable[AstNode, NimNode]()

  for i, inOutVar in io:
    if inOutVar.kind != Input:
      if inOutVar.kind in {Output, LVal}:
        let sym = walkASTGeneric(inOutVar, visitedNodes, body)
        sym.expectKind nnkIdent
        if resultTy.kind == nnkTupleTy:
          body.add newAssignment(
            nnkDotExpr.newTree(
              newIdentNode"result",
              procIdents[i]
            ),
            sym
          )
        else:
          body.add newAssignment(
            newIdentNode"result",
            sym
          )
      else:
        let expression = walkASTGeneric(inOutVar, visitedNodes, body)
        if resultTy.kind == nnkTupleTy:
          body.add newAssignment(
            nnkDotExpr.newTree(
              newIdentNode"result",
              procIdents[i]
            ),
            expression
          )
        else:
          body.add newAssignment(
            newIdentNode"result",
            expression
          )

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
  baz += b
  buzz += b

compile([a, b, c, bar, baz, buzz]):
  proc foobar[T](a, b, c: T): tuple[bar, baz, buzz: T]

let (pim, pam, poum) = foobar(1, 2, 3)

echo pim # 12
echo pam # 20
echo poum # 10020
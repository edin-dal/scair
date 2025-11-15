package scair.interpreter

import scair.ir.*

type InterpreterDialect = Seq[OpImpl[? <: Operation]]

val allInterpreterDialects: Seq[InterpreterDialect] =
  Seq(
    InterpreterFuncDialect,
    InterpreterArithDialect,
    InterpreterMemrefDialect
  )

package scair.interpreter

import scair.ir.*

type InterpreterDialect =
  Seq[OpImpl[? <: Operation] | OpTerminatorImpl[? <: Operation]]

val allInterpreterDialects: Seq[InterpreterDialect] =
  Seq(
    InterpreterFuncDialect,
    InterpreterArithDialect,
    InterpreterMemrefDialect,
    InterpreterScfDialect,
    InterpreterLLVMDialect,
  )

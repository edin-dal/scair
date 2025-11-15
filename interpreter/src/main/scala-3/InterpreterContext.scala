package scair.interpreter

import scair.ir.*

import scala.collection.mutable

// definition of all dialects in interpreter context
val interpreterContext = InterpreterContext(
      Seq(
        InterpreterFuncDialect,
        InterpreterArithDialect,
        InterpreterMemrefDialect
      ),
      mutable.Map()
)

// global interpreter context to hold all registered dialects and their implementations
// may have future uses
case class InterpreterContext(
    val interpreterDialects: Seq[InterpreterDialect],
    val implementationCtx: mutable.Map[Class[? <: Operation], OpImpl[
      ? <: Operation
    ]]
):
  // registering all dialects
  def registerInterpreterDialects(): Unit =
    for dialect <- interpreterDialects do registerImplementations(dialect)

  // registers all implementations in a dialect
  def registerImplementations(dialect: InterpreterDialect): Unit =
    for impl <- dialect.implementations do
      implementationCtx.put(impl.opType, impl)

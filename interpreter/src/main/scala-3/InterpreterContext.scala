package scair.tools

import scair.ir.*

import scala.collection.mutable

// global interpreter context to hold all registered dialects and their implementations
// may have future uses
case class InterpreterContext(
    val interpreterDialects: Seq[InterpreterDialect],
    val implementationCtx: mutable.Map[Class[? <: Operation], OpImpl[
      ? <: Operation
    ]]
):
  // registering all dialects
  def registerInterpreterDialects(dialects: Seq[InterpreterDialect]): Unit =
    for dialect <- dialects do registerImplementations(dialect)

  // registers all implementations in a dialect
  def registerImplementations(dialect: InterpreterDialect): Unit =
    for impl <- dialect.implementations do
      implementationCtx.put(impl.opType, impl)

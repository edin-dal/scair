package scair.tools

import scala.collection.mutable
import scair.ir.*

final case class InterpreterDialect(
    val implementations: Seq[OpImpl[? <: Operation]]
)

case class InterpreterContext(
    val interpreterDialects: Seq[InterpreterDialect],
    val implementationCtx: mutable.Map[Class[? <: Operation], OpImpl[? <: Operation]]
):
    def registerInterpreterDialects(dialects: Seq[InterpreterDialect]): Unit =
        println("reached")
        for dialect <- dialects do
            registerImplementations(dialect)

    def registerImplementations(dialect: InterpreterDialect): Unit =
        for impl <- dialect.implementations do
            implementationCtx.put(impl.opType, impl)




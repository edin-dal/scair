package scair.tools

import scala.collection.mutable
import scair.ir.*

case class InterpreterContext(
    val interpreterDialects: Seq[InterpreterDialect],
    val implementationCtx: mutable.Map[? <: Operation, OpImpl[? <: Operation]]
):
    def registerImplementations(dialects: InterpreterDialect): Unit =
        for impl <- dialects.implementations do
            None




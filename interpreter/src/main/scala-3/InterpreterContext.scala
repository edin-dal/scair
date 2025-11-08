package scair.tools

import scala.collection.mutable
import scair.ir.*

case class InterpreterContext(
    val interpreterDialects: Seq[InterpreterDialects],
    val implementationCtx: mutable.Map[Operation, OperationImpl]
):
    def registerImplementations(dialects: InterpreterDialects): Unit =
        for impl <- dialects.implementations do
            None




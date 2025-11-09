package scair.tools

import scala.collection.mutable
import scair.ir.*

type OperationImpl = (Operation, RuntimeCtx) => Unit

final case class InterpreterDialects(
    val implementations: Seq[OperationImpl]
)

case class InterpreterContext(
    val interpreterDialects: Seq[InterpreterDialects],
    val implementationCtx: mutable.Map[Operation, OperationImpl]
):
    def registerImplementations(dialects: InterpreterDialects): Unit =
        for impl <- dialects.implementations do
            None




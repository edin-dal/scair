package scair.tools

import scala.collection.mutable
import scair.ir.*

case class OpImpl(opType: Class[? <: Operation], run: (Operation, RuntimeCtx) => Unit)

final case class InterpreterDialect(
    val implementations: Seq[OpImpl]
)

inline def summonImplementations(impls: Seq[OpImpl]): InterpreterDialect = 
    new InterpreterDialect(impls)

case class InterpreterContext(
    val interpreterDialects: Seq[InterpreterDialect],
    val implementationCtx: mutable.Map[? <: Operation, OpImpl]
):
    def registerImplementations(dialects: InterpreterDialect): Unit =
        for impl <- dialects.implementations do
            None




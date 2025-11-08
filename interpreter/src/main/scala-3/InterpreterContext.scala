package scair.tools

import scala.collection.mutable

final case class InterpreterDialects(
    val implementations: Seq[Any]
)

val allInterpreterDialects: Seq[InterpreterDialects] = Seq(
    InterpreterDialects(Seq())
)

case class InterpreterContext(
    val interpreterDialects: Seq[InterpreterDialects],
    val implementationCtx: mutable.Map[String,Any]
):
    def registerImplementations(): Seq[Any] = Seq()



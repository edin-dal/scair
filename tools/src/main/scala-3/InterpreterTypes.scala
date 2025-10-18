package scair.tools

import scair.ir.*
import scala.collection.mutable
import scala.collection.mutable.ListBuffer

// note: arguments in function treated like variables in an environment
// when interpreting function, create new nested context with args as vars
// when interpreting variable, look up in current context, if not found look up in parent context
// when finished interpreting function definition, add function context to that level of ctx
class InterpreterCtx (
    val vars: mutable.Map[Operation, Attribute],
    val memory: mutable.Map[Int, Attribute],
    val funcs: ListBuffer[FunctionCtx],
    var result: Attribute
)

case class FunctionCtx(
  name: String,
  func_ctx: InterpreterCtx,
  body: Block
)



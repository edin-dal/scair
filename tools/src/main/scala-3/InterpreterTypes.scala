package scair.tools

import scair.ir.*
import scair.dialects.func

// note: arguments in function treated like variables in an environment
// when interpreting function, create new nested context with args as vars
// when interpreting variable, look up in current context, if not found look up in parent context
class InterpreterCtx (
    vars: Map[Int, Attribute],
    memory: Map[Int, Attribute],
    funcs: List[FunctionCtx]
)

case class FunctionCtx(
  name: String,
  operands: Map[Int, Attribute],
  parent: Option[FunctionCtx],
  body: func.Func,
  result: Option[Attribute] = None
)



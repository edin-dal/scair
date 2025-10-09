package scair.tools

import scair.ir.*
import scair.dialects.func

// note: arguments in function treated like variables in an environment
// when interpreting function, create new nested context with args as vars
// when interpreting variable, look up in current context, if not found look up in parent context
// TODO: investigate how functions are stored in module and how to begin evaluating functions
class InterpreterCtx (
    // TODO: change these
    vars: Map[String, Attribute],
    memory: Map[Int, Attribute],
    parent: Option[InterpreterCtx],
    funcs: Map[String, func.Func]
)



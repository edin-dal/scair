package scair.tools

import scair.ir.*
import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scair.dialects.func

// note: arguments in function treated like variables in an environment
// when interpreting function, create new nested context with args as vars
// when interpreting variable, look up in current context, if not found look up in parent context
// when finished interpreting function definition, add function context to that level of ctx
class InterpreterCtx (
    val vars: mutable.Map[Operation, Attribute],
    val memory: ListBuffer[Attribute],
    val funcs: ListBuffer[FunctionCtx],
    var result: Option[Attribute] = None
) {
    // creates independent context
    def clone_ctx(): InterpreterCtx = {
        InterpreterCtx(
            mutable.Map() ++ this.vars,
            ListBuffer() ++ this.memory,
            ListBuffer() ++ this.funcs,
            None
        )
    }

    def add_func_ctx(function: func.Func): Unit = {
        val func_ctx = FunctionCtx(
            name = function.sym_name,
            saved_ctx = this.clone_ctx(),
            body = function.body.blocks.head
        )
        this.funcs.append(func_ctx)
    }
}

case class FunctionCtx(
  name: String,
  saved_ctx: InterpreterCtx,
  body: Block
)



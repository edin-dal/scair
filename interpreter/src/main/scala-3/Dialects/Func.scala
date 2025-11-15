package scair.tools

import scair.dialects.func

// assume one return value for now
object run_return extends OpImpl[func.Return]:
  def run(op: func.Return, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
    ctx.result = Some(op._operands.head)

object run_call extends OpImpl[func.Call]:
  def run(op: func.Call, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
    for func_ctx <- ctx.funcs do
      if func_ctx.name == op.callee.rootRef.stringLiteral then
        val new_ctx = func_ctx.saved_ctx.deep_clone_ctx()
        for op <- func_ctx.body.operations do
          interpreter.interpret_op(op, new_ctx) // create clone so function can run without modifying saved context
        ctx.vars.put(op._results.head, new_ctx.result.get) // assuming one return value for now

        


val InterpreterFuncDialect = summonImplementations(
  Seq(
    run_return,
    run_call
  )
)
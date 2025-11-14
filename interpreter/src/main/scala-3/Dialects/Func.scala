package scair.tools

import scair.dialects.func

object run_return extends OpImpl[func.Return]:
  def run(op: func.Return, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
    val return_results = for op <- op.operands yield lookup_op(op, ctx)
    // if one singular result, wrap as Some(result), wrap multiple as Some(Seq[result]) and None if no result
    if return_results.length == 1 then
      ctx.result = Some(return_results.head)
    else if return_results.length == 0 then ctx.result = None
    else ctx.result = Some(return_results)

object run_call extends OpImpl[func.Call]:
  def run(op: func.Call, interpreter: Interpreter, ctx: RuntimeCtx): Unit = 
    for func_ctx <- ctx.funcs do
      if func_ctx.name == op.callee.rootRef.stringLiteral then
        interpreter.interpret_block_or_op(func_ctx.body, func_ctx.saved_ctx)

val InterpreterFuncDialect = summonImplementations(
  Seq(
    run_return
  )
)
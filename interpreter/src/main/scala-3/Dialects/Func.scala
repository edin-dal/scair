package scair.tools

import scair.dialects.func

def run_return(op: func.Return, ctx: RuntimeCtx): Unit =
  val return_results = for op <- op.operands yield lookup_op(op, ctx)
  // if one singular result, wrap as Some(result), wrap multiple as Some(Seq[result]) and None if no result
  if return_results.length == 1 then
    ctx.result = Some(return_results.head)
  else if return_results.length == 0 then ctx.result = None
  else ctx.result = Some(return_results)

def run_function_call(op: func.Call, ctx: RuntimeCtx): Unit = 0
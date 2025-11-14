package scair.tools

import scair.dialects.func

object run_return extends OpImpl[func.Return]:
  def run(op: func.Return, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
    val result_ops = for op <- op.operands yield op
    ctx.result = Some(result_ops)

object run_call extends OpImpl[func.Call]:
  def run(op: func.Call, interpreter: Interpreter, ctx: RuntimeCtx): Unit = 0
//     for func_ctx <- ctx.funcs do
//       if func_ctx.name == op.callee.rootRef.stringLiteral then
//         val new_ctx = func_ctx.saved_ctx.deep_clone_ctx()
//         for op <- func_ctx.body.operations do
//           interpreter.interpret_op(op, new_ctx) // create clone so function can run without modifying saved context

        


val InterpreterFuncDialect = summonImplementations(
  Seq(
    run_return,
    run_call
  )
)
package scair.tools

import scair.dialects.func
import scair.ir.Result
import scair.dialects.builtin.SymbolRefAttr

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

object run_function extends OpImpl[func.Func]:
  def run(op: func.Func, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
    if op.sym_name.stringLiteral == "main" then
      val main_ctx = FunctionCtx(
        name = op.sym_name,
        body = op.body.blocks.head,
        saved_ctx = ctx
      )
      ctx.funcs.append(main_ctx)
      val new_call = func.Call(
        callee = SymbolRefAttr(op.sym_name),
        _operands = Seq(),
        _results = op.function_type.outputs.map(res => Result(res))
      )
      run_call.run(new_call, interpreter, ctx)
    else
      ctx.add_func_ctx(op)

        


val InterpreterFuncDialect = summonImplementations(
  Seq(
    run_return,
    run_call,
    run_function
  )
)
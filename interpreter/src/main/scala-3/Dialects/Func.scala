package scair.interpreter

import scair.dialects.builtin.SymbolRefAttr
import scair.dialects.func
import scair.ir.Result

// assume one return value for now
object run_return extends OpImpl[func.Return]:

  def run(op: func.Return, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
    ctx.result = Some(interpreter.lookup_op(op._operands.head, ctx))

object run_call extends OpImpl[func.Call]:

  def run(op: func.Call, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
    for func_ctx <- ctx.funcs do
      if func_ctx.name == op.callee.rootRef.stringLiteral then
        val new_ctx = func_ctx.saved_ctx.deep_clone_ctx()
        for op <- func_ctx.body.operations do
          interpreter.interpret_op(
            op,
            new_ctx
          ) // create clone so function can run without modifying saved context
        ctx.vars.put(
          op._results.head,
          new_ctx.result.getOrElse(None)
        ) // assuming one return value for now

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
      // should it be external call like xDSL?
      run_call.run(new_call, interpreter, ctx)
      // get return value from main call and add to context
      val return_result = interpreter.lookup_op(new_call._results.head, ctx)
      ctx.result = Some(return_result)
    else ctx.add_func_ctx(op)

val InterpreterFuncDialect: InterpreterDialect =
  Seq(
    run_return,
    run_call,
    run_function
  )

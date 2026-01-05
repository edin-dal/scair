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
    // if call for print, print
    // later there may be a print operation instead
    if op.callee.rootRef.stringLiteral == "print" then
      val print_value = interpreter.lookup_op(op._operands.head, ctx)
      interpreter.interpreter_print(print_value)
    else
      val new_ctx = ctx.push_scope()
      for operands <- op._operands do
        val operand_value = interpreter.lookup_op(operands, ctx)
        new_ctx.scopedDict.update(operands, operand_value)
      val callee = ctx.symbols.get(op.callee.rootRef.stringLiteral).get
          .asInstanceOf[func.Func] // presume func if called
      for op <- callee.body.blocks.head.operations do
        interpreter.interpret_op(
          op,
          new_ctx,
        ) // create clone so function can run without modifying saved context
      ctx.scopedDict.update(
        op._results.head,
        new_ctx.result.getOrElse(None),
      ) // assuming one return value for now

object run_function extends OpImpl[func.Func]:

  def run(op: func.Func, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
    // add function to symbol table
    ctx.symbols.put(op.sym_name.stringLiteral, op)

    // if main function, call it immediately
    if op.sym_name.stringLiteral == "main" then
      val new_call = func.Call(
        callee = SymbolRefAttr(op.sym_name),
        _operands = Seq(),
        _results = op.function_type.outputs.map(res => Result(res)),
      )
      // should it be external call like xDSL?
      run_call.run(new_call, interpreter, ctx)
      // get return value from main call and add to context
      val return_result = interpreter.lookup_op(new_call._results.head, ctx)
      ctx.result = Some(return_result)

val InterpreterFuncDialect: InterpreterDialect =
  Seq(
    run_return,
    run_call,
    run_function,
  )

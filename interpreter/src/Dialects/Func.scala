package scair.interpreter

import scair.dialects.builtin.SymbolRefAttr
import scair.dialects.func
import scair.ir.Result

object run_return extends OpImpl[func.Return]:

  def compute(
      op: func.Return,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Seq[Any],
  ): Seq[Any] =
    return args

object run_call extends OpImpl[func.Call]:

  def compute(
      op: func.Call,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Seq[Any],
  ): Seq[Any] =
    // if call for print, print
    // later there may be a print operation instead
    if op.callee.rootRef.stringLiteral == "print" then
      val print_value = interpreter.lookup_op(op._operands.head, ctx)
      interpreter.interpreter_print(print_value)
      Seq()
    else
      interpreter.call_op(op.callee.rootRef.stringLiteral, ctx, args)

object run_function extends OpImpl[func.Func]:

  // only needed for main
  def compute(
      op: func.Func,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Seq[Any],
  ): Seq[Any] =
    val new_call = func.Call(
      callee = SymbolRefAttr(op.sym_name),
      _operands = Seq(),
      _results = op.function_type.outputs.map(res => Result(res)),
    )
    val results = interpreter.run_op(new_call, ctx, Seq())

    if new_call._results.nonEmpty then
      results
    else Seq()

val InterpreterFuncDialect: InterpreterDialect =
  Seq(
    run_return,
    run_call,
    run_function,
  )

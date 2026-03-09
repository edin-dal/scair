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
    args match
      case Seq(v) => ctx.result = Seq(v)
      case Seq()  => ctx.result = Seq()
      case _      =>
        throw new Exception("Expected args to be a Seq of values, got: " + args)
    Seq()

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
      // new context with new scoped dict containing function operands but shared symbol table
      val new_ctx = ctx.push_scope(op.callee.rootRef.stringLiteral)
      interpreter.scopes += new_ctx.scopedDict
      val callee = interpreter.symbolTable.get(op.callee.rootRef.stringLiteral)
        .get.asInstanceOf[func.Func] // presume func if called

      // adds function argument to scoped dict since they have the reference
      // that will be matched during lookup and maps it to the defined operand value
      // TODO: check over
      for (operand, param) <- args.zip(callee.body.blocks.head.arguments)
      do new_ctx.scopedDict.update(param, operand)

      for op <- callee.body.blocks.head.operations do
        interpreter.interpret_op(
          op,
          new_ctx,
        ) // create clone so function can run without modifying saved context

      if op._results.nonEmpty then new_ctx.result else Seq()

object run_function extends OpImpl[func.Func]:

  // only needed for main
  def compute(
      op: func.Func,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Seq[Any],
  ): Seq[Any] =

    // if main function, call it immediately
    if op.sym_name.stringLiteral == "main" then
      val new_call = func.Call(
        callee = SymbolRefAttr(op.sym_name),
        _operands = Seq(),
        _results = op.function_type.outputs.map(res => Result(res)),
      )
      run_call.run(new_call, interpreter, ctx)

      if new_call._results.nonEmpty then
        new_call._results.map(res => interpreter.lookup_op(res, ctx))
      else Seq()
    else Seq()

val InterpreterFuncDialect: InterpreterDialect =
  Seq(
    run_return,
    run_call,
    run_function,
  )

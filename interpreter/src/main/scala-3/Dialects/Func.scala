package scair.interpreter

import scair.dialects.builtin.SymbolRefAttr
import scair.dialects.func
import scair.ir.Result

import scala.collection.mutable

// assume one return value for now
object run_return extends OpImpl[func.Return]:

  def compute(
      op: func.Return,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Seq[Any],
  ): Any =
    args match
      case Seq(v) => ctx.result = Some(v)
      case _      => ctx.result = Some(args)

object run_call extends OpImpl[func.Call]:

  def compute(
      op: func.Call,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Seq[Any],
  ): Any =
    // if call for print, print
    // later there may be a print operation instead
    if op.callee.rootRef.stringLiteral == "print" then
      val print_value = interpreter.lookup_op(op._operands.head, ctx)
      interpreter.interpreter_print(print_value)
    else
      // new context with new scoped dict containing function operands but shared symbol table
      val new_ctx = ctx.push_scope(op.callee.rootRef.stringLiteral)
      interpreter.scopes ++ Seq(new_ctx.scopedDict)
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

      if op._results.nonEmpty then return new_ctx.result.getOrElse(None)

object run_function extends OpImpl[func.Func]:

  // only needed for main
  def compute(
      op: func.Func,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Seq[Any],
  ): Any =

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

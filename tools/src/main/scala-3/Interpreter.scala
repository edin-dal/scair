package scair.tools

import scair.dialects.arith
import scair.dialects.builtin.*
import scair.dialects.func
import scair.dialects.memref
import scair.ir.*

// INTERPRETER CLASS
class Interpreter extends ArithmeticEvaluator with MemoryHandler:

  // keeping buffer function for extensibility
  def interpret(block: Block, ctx: InterpreterCtx): Option[Any] =
    for op <- block.operations do interpret_op(op, ctx)
    ctx.result

  // main operation interpretation function
  def interpret_op(op: Operation, ctx: InterpreterCtx): Unit =
    op match
      case func_op: func.Func =>
        interpret_function(func_op, ctx)

      // Return Operation
      case return_op: func.Return =>
        val return_results =
          for op <- return_op._operands yield lookup_op(op, ctx)
        if return_results.length == 1 then
          ctx.result = Some(return_results.head)
        else if return_results.length == 0 then ctx.result = None
        else ctx.result = Some(return_results)

      // Constant Operation
      case constant_op: arith.Constant =>
        val value = interpret_constant(constant_op)
        ctx.vars.put(constant_op.result, value)

      // Binary Operations
      case addI_op: arith.AddI =>
        ctx.vars.put(
          addI_op.result,
          interpret_bin_op(addI_op.lhs, addI_op.rhs, ctx)(_ + _)
        )
      case subI_op: arith.SubI =>
        ctx.vars.put(
          subI_op.result,
          interpret_bin_op(subI_op.lhs, subI_op.rhs, ctx)(_ - _)
        )
      case mulI_op: arith.MulI =>
        ctx.vars.put(
          mulI_op.result,
          interpret_bin_op(mulI_op.lhs, mulI_op.rhs, ctx)(_ * _)
        )
      case div_op: arith.DivSI =>
        ctx.vars.put(
          div_op.result,
          interpret_bin_op(div_op.lhs, div_op.rhs, ctx)(_ / _)
        )
      case div_op: arith.DivUI =>
        ctx.vars.put(
          div_op.result,
          interpret_bin_op(div_op.lhs, div_op.rhs, ctx)(_ / _)
        )
      case andI_op: arith.AndI =>
        ctx.vars.put(
          andI_op.result,
          interpret_bin_op(andI_op.lhs, andI_op.rhs, ctx)(_ & _)
        )
      case orI_op: arith.OrI =>
        ctx.vars.put(
          orI_op.result,
          interpret_bin_op(orI_op.lhs, orI_op.rhs, ctx)(_ | _)
        )
      case xorI_op: arith.XOrI =>
        ctx.vars.put(
          xorI_op.result,
          interpret_bin_op(xorI_op.lhs, xorI_op.rhs, ctx)(_ ^ _)
        )

      // Shift Operations
      case shli_op: arith.ShLI =>
        ctx.vars.put(
          shli_op.result,
          interpret_bin_op(shli_op.lhs, shli_op.rhs, ctx)(_ << _)
        )
      case shrsi_op: arith.ShRSI =>
        ctx.vars.put(
          shrsi_op.result,
          interpret_bin_op(shrsi_op.lhs, shrsi_op.rhs, ctx)(_ >> _)
        )
      case shrui_op: arith.ShRUI =>
        ctx.vars.put(
          shrui_op.result,
          interpret_bin_op(shrui_op.lhs, shrui_op.rhs, ctx)(_ >>> _)
        )

      // Comparison Operation
      case cmpi_op: arith.CmpI =>
        ctx.vars.put(
          cmpi_op.result,
          interpret_cmp_op(
            cmpi_op.lhs,
            cmpi_op.rhs,
            cmpi_op.predicate.value.toInt,
            ctx
          )
        )

      // Select Operation
      case select_op: arith.SelectOp =>
        val cond_val = lookup_op(select_op.condition, ctx)
        cond_val match
          case condOp: Int =>
            condOp match
              case 0 =>
                ctx.vars.put(
                  select_op.result,
                  lookup_op(select_op.falseValue, ctx)
                )
              case 1 =>
                ctx.vars.put(
                  select_op.result,
                  lookup_op(select_op.trueValue, ctx)
                )
              case _ => throw new Exception("Select condition must be 0 or 1")
          case _ =>
            throw new Exception("Select condition must be an integer attribute")

      // Memory operations
      case alloc_op: memref.Alloc =>
        allocate_memory(alloc_op, ctx)
      case load_op: memref.Load =>
        load_memory(load_op, ctx)
      case store_op: memref.Store =>
        store_memory(store_op, ctx)
      case _ => throw new Exception("Unsupported operation when interpreting")

  def interpret_block_or_op(
      value: Operation | Block,
      ctx: InterpreterCtx
  ): Unit =
    value match
      case op: Operation => interpret_op(op, ctx)
      case block: Block  =>
        for op <- block.operations do interpret_op(op, ctx)

  def interpret_function(function: func.Func, ctx: InterpreterCtx): Unit =
    // if main, interpret it immediately by creating a call operation and evaluating it
    if function.sym_name.stringLiteral == "main" then
      val main_ctx = FunctionCtx(
        name = function.sym_name,
        body = function.body.blocks.head,
        saved_ctx = ctx
      )
      ctx.funcs.append(main_ctx)
      val new_call = func.Call(
        callee = SymbolRefAttr(function.sym_name),
        _operands = Seq(),
        _results = function.function_type.outputs.map(res => Result(res))
      )
      interpret_call(new_call, ctx)
    else
      // function definition; no call, add function and current running context functionCtx to interpreter context
      ctx.add_func_ctx(function)

  def interpret_call(call_op: func.Call, ctx: InterpreterCtx): Unit =
    for func_ctx <- ctx.funcs do
      if func_ctx.name == call_op.callee.rootRef.stringLiteral then
        interpret_block_or_op(func_ctx.body, func_ctx.saved_ctx)

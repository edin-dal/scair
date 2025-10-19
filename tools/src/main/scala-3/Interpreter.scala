package scair.tools

import scair.ir.*
import scair.dialects.arith
import scair.dialects.func
import scair.dialects.builtin.*


// TODO: do flow analysis to find correct return op
class Interpreter {

  // TODO: only passing in block for now, may need to generalise for list of blocks later
  // keeping buffer function for extensibility
  def interpret(block: Block, ctx: InterpreterCtx): Option[Attribute] = {
    for (op <- block.operations) {
      interpret_op(op, ctx)
    }
    ctx.result
  }

  def interpret_call(call_op: func.Call, ctx: InterpreterCtx): Unit = {
    for (func_ctx <- ctx.funcs) {
      if (func_ctx.name == call_op.callee.rootRef.stringLiteral) {
        interpret_block(func_ctx.body, ctx)
      }
    }
  }

  def interpret_op(op: Operation, ctx: InterpreterCtx): Unit = {
    op match {

      case func_op: func.Func =>
        interpret_function(func_op, ctx)

      // Function Return
      case return_op: func.Return =>
        // TODO: multiple return values
        ctx.result = Some(lookup_op(return_op.operands.head.owner.get, ctx))


      // Constants
      case constant: arith.Constant =>
        ctx.vars.put(constant, constant.value)

      // TODO: handling block arguments for all arithmetic operations
      // Binary Operations
      case addIOp: arith.AddI => 
        interpret_bin_op(addIOp, addIOp.lhs, addIOp.rhs, "AddI", ctx)(_ + _)
      case subIOp: arith.SubI => 
        interpret_bin_op(subIOp, subIOp.lhs, subIOp.rhs, "SubI", ctx)(_ - _)
      case mulIOp: arith.MulI => 
        interpret_bin_op(mulIOp, mulIOp.lhs, mulIOp.rhs, "MulI", ctx)(_ * _)
      case divOp: arith.DivSI => 
        interpret_bin_op(divOp, divOp.lhs, divOp.rhs, "DivSI", ctx)(_ / _)
      case divOp: arith.DivUI =>
        interpret_bin_op(divOp, divOp.lhs, divOp.rhs, "DivUI", ctx)(_ / _)
      case andIOp: arith.AndI =>
        interpret_bin_op(andIOp, andIOp.lhs, andIOp.rhs, "AndI", ctx)(_ & _)
      case orIOp: arith.OrI =>
        interpret_bin_op(orIOp, orIOp.lhs, orIOp.rhs, "OrI", ctx)(_ | _)
      case xorIOp: arith.XOrI =>
        interpret_bin_op(xorIOp, xorIOp.lhs, xorIOp.rhs, "XorI", ctx)(_ ^ _)

      // Unary Operations 

      // Shift Operations
      case shliOp: arith.ShLI =>
        interpret_bin_op(shliOp, shliOp.lhs, shliOp.rhs, "ShLI", ctx)(_ << _)
      case shrsiOp: arith.ShRSI =>
        interpret_bin_op(shrsiOp, shrsiOp.lhs, shrsiOp.rhs, "ShRSI", ctx)(_ >> _)
      case shruiOp: arith.ShRUI =>
        interpret_bin_op(shruiOp, shruiOp.lhs, shruiOp.rhs, "ShRUI", ctx)(_ >>> _)

      // Comparison Operation
      case cmpiOp: arith.CmpI =>
        interpret_cmp_op(cmpiOp.lhs, cmpiOp.rhs, cmpiOp.predicate.value.toInt, ctx)

      // Select Operation
      case selectOp: arith.SelectOp =>
        // (interpret_block_or_op(selectOp.condition.owner.get, ctx)) match {
        //   case condOp: IntegerAttr => 
        //     condOp.value.value.toInt match {
        //       case 0 => interpret_block_or_op(selectOp.falseValue.owner.get, ctx)
        //       case 1 => interpret_block_or_op(selectOp.trueValue.owner.get, ctx)
        //       case _ => throw new Exception("Select condition must be 0 or 1")
        //     }
        //   case _ => throw new Exception("Select condition must be an integer attribute")
        //   }
      case _ => throw new Exception("Unsupported operation")
    }
  }

  // interpret block by iterating and evaluating each operation in it 
  def interpret_block(block: Block, ctx: InterpreterCtx): Unit = {
    for (op <- block.operations) {
      interpret_op(op, ctx)
    }
  }

  // TODO: maybe make sure bitwidth is ok?
  def interpret_bin_op(op: Operation, lhs: Value[Attribute], rhs: Value[Attribute], name: String, ctx: InterpreterCtx)
  (combine: (Int, Int) => Int): Unit = {
    val lval = lookup_op(lhs.owner.get, ctx)
    val rval = lookup_op(rhs.owner.get, ctx)
    (lval, rval) match {
      case (li: IntegerAttr, ri: IntegerAttr) =>
        val result = IntegerAttr(IntData(combine(li.value.value.toInt, ri.value.value.toInt)), li.typ)
        ctx.vars.put(op, result)
      case _ => throw new Exception(s"Unsupported operand types for $name")
    }
  }

  def interpret_cmp_op(lhs: Value[Attribute], rhs: Value[Attribute], predicate: Int, ctx: InterpreterCtx): Unit = {
    // (lhs.owner.get, rhs.owner.get) match {
    //   case (l: Operation, r: Operation) =>
    //     (interpret_op(l, ctx), interpret_op(r, ctx)) match {
    //       case (li: IntegerAttr, ri: IntegerAttr) =>
    //         val lval = li.value.value
    //         val rval = ri.value.value
    //         predicate match {
    //           case 0 => // EQ
    //             if (lval == rval) IntegerAttr(IntData(1), I1) else IntegerAttr(IntData(0), I1)
    //           case 1 => // NE
    //             if (lval != rval) IntegerAttr(IntData(1), I1) else IntegerAttr(IntData(0), I1)
    //           case 2 | 6 => // SLT and ULT, assume both numbers are already converted accoringly
    //             if (lval < rval) IntegerAttr(IntData(1), I1) else IntegerAttr(IntData(0), I1)
    //           case 3 | 7 => // SLE and ULE
    //             if (lval <= rval) IntegerAttr(IntData(1), I1) else IntegerAttr(IntData(0), I1)
    //           case 4 | 8 => // SGT and UGT
    //             if (lval > rval) IntegerAttr(IntData(1), I1) else IntegerAttr(IntData(0), I1)
    //           case 5 | 9 => // SGE and UGE
    //             if (lval >= rval) IntegerAttr(IntData(1), I1) else IntegerAttr(IntData(0), I1)
    //           case _ => throw new Exception("Unknown comparison predicate")
    //         }
    //       case other => throw new Exception("Unsupported operand types for cmpi, got: $other")
    //     }
    //   case other => throw new Exception("Expected operations as operands for cmpi, got: $other")
    // }
  }

  def interpret_block_or_op(value: Operation | Block, ctx: InterpreterCtx): Unit = {
    value match {
      case op: Operation => interpret_op(op, ctx)
      case block: Block => interpret_block(block, ctx)
    }
  }

  def interpret_function(function: func.Func, ctx: InterpreterCtx): Unit = {
    // if main, interpret it immediately by creating a call operation and evaluating it
    if (function.sym_name.stringLiteral == "main") {
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
    } else {
      // function definition; no call, add function and current running context functionCtx to interpreter context
      ctx.add_func_ctx(function)
    }
  }
}

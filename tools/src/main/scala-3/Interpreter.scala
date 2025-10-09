package scair.tools

import scair.ir.*
import scair.dialects.arith
import scair.dialects.func
import scair.dialects.builtin.*


// TODO: do flow analysis to find correct return op
class Interpreter {

  // TODO: generic interpret op, probably will become block/function evaluator soon
  def interpret(op: Operation): Attribute = {
    // presuming main function for now
    val interpreterCtx = new InterpreterCtx(Map(), None, Map())
    op match {
      case func_op: func.Func =>
        val return_op = func_op.body.blocks.head.operations.last
        interpret_op(return_op, interpreterCtx)
      case other => throw new Exception("Expecting function")
    }

    
  }

  def interpret_op(op: Operation, ctx: InterpreterCtx): Attribute = {
    op match {

      // Function Return
      // TODO: multiple return vals, environments
      case funcReturnOp: func.Return =>
        val return_root = funcReturnOp.operands.head.owner.getOrElse(0).asInstanceOf[Operation]
        interpret_op(return_root, ctx)

      // Constants
      // TODO: vectors?
      case constant: arith.Constant => constant.value

      // TODO: handling block arguments for all arithmetic operations
      // Binary Operations
      case addIOp: arith.AddI => 
        interpret_bin_op(addIOp.lhs, addIOp.rhs, "AddI", ctx)(_ + _)
      case subIOp: arith.SubI => 
        interpret_bin_op(subIOp.lhs, subIOp.rhs, "SubI", ctx)(_ - _)
      case mulIOp: arith.MulI => 
        interpret_bin_op(mulIOp.lhs, mulIOp.rhs, "MulI", ctx)(_ * _)
      case divOp: arith.DivSI => 
        interpret_bin_op(divOp.lhs, divOp.rhs, "DivSI", ctx)(_ / _)
      case divOp: arith.DivUI =>
        interpret_bin_op(divOp.lhs, divOp.rhs, "DivUI", ctx)(_ / _)
      case andIOp: arith.AndI =>
        interpret_bin_op(andIOp.lhs, andIOp.rhs, "AndI", ctx)(_ & _)
      case orIOp: arith.OrI =>
        interpret_bin_op(orIOp.lhs, orIOp.rhs, "OrI", ctx)(_ | _)
      case xorIOp: arith.XOrI =>
        interpret_bin_op(xorIOp.lhs, xorIOp.rhs, "XorI", ctx)(_ ^ _)

      // Unary Operations 

      // Shift Operations
      case shliOp: arith.ShLI =>
        interpret_bin_op(shliOp.lhs, shliOp.rhs, "ShLI", ctx)(_ << _)
      case shrsiOp: arith.ShRSI =>
        interpret_bin_op(shrsiOp.lhs, shrsiOp.rhs, "ShRSI", ctx)(_ >> _)
      case shruiOp: arith.ShRUI =>
        interpret_bin_op(shruiOp.lhs, shruiOp.rhs, "ShRUI", ctx)(_ >>> _)

      // Comparison Operation
      case cmpiOp: arith.CmpI =>
        interpret_cmp_op(cmpiOp.lhs, cmpiOp.rhs, cmpiOp.predicate.value.toInt, ctx)

      // Select Operation
      case selectOp: arith.SelectOp =>
        (interpret_block_or_op(selectOp.condition.owner.get, ctx)) match {
          case condOp: IntegerAttr => 
            condOp.value.value.toInt match {
              case 0 => interpret_block_or_op(selectOp.falseValue.owner.get, ctx)
              case 1 => interpret_block_or_op(selectOp.trueValue.owner.get, ctx)
              case _ => throw new Exception("Select condition must be 0 or 1")
            }
          case _ => throw new Exception("Select condition must be an integer attribute")
        }
      case _ => throw new Exception("Unsupported operation")
    }
}

  // TODO: maybe make sure bitwidth is ok?
  def interpret_bin_op(lhs: Value[Attribute], rhs: Value[Attribute], name: String, ctx: InterpreterCtx)
  (combine: (Int, Int) => Int): Attribute = {
    (lhs.owner.get, rhs.owner.get) match {
      case (l: Operation, r: Operation) =>
        (interpret_op(l, ctx), interpret_op(r, ctx)) match {
          case (li: IntegerAttr, ri: IntegerAttr) =>
            val lval = li.value.value.toInt
            val rval = ri.value.value.toInt
            IntegerAttr(IntData(combine(lval, rval)), li.typ)
          case other => throw new Exception("Unsupported operand types for $name, got: $other")
        }
      case other => throw new Exception("Expected operations as operands for $name, got: $other")
    }
  }

  def interpret_cmp_op(lhs: Value[Attribute], rhs: Value[Attribute], predicate: Int, ctx: InterpreterCtx): Attribute = {
    (lhs.owner.get, rhs.owner.get) match {
      case (l: Operation, r: Operation) =>
        (interpret_op(l, ctx), interpret_op(r, ctx)) match {
          case (li: IntegerAttr, ri: IntegerAttr) =>
            val lval = li.value.value
            val rval = ri.value.value
            predicate match {
              case 0 => // EQ
                if (lval == rval) IntegerAttr(IntData(1), I1) else IntegerAttr(IntData(0), I1)
              case 1 => // NE
                if (lval != rval) IntegerAttr(IntData(1), I1) else IntegerAttr(IntData(0), I1)
              case 2 | 6 => // SLT and ULT, assume both numbers are already converted accoringly
                if (lval < rval) IntegerAttr(IntData(1), I1) else IntegerAttr(IntData(0), I1)
              case 3 | 7 => // SLE and ULE
                if (lval <= rval) IntegerAttr(IntData(1), I1) else IntegerAttr(IntData(0), I1)
              case 4 | 8 => // SGT and UGT
                if (lval > rval) IntegerAttr(IntData(1), I1) else IntegerAttr(IntData(0), I1)
              case 5 | 9 => // SGE and UGE
                if (lval >= rval) IntegerAttr(IntData(1), I1) else IntegerAttr(IntData(0), I1)
              case _ => throw new Exception("Unknown comparison predicate")
            }
          case other => throw new Exception("Unsupported operand types for cmpi, got: $other")
        }
      case other => throw new Exception("Expected operations as operands for cmpi, got: $other")
    }
  }

  def interpret_block_or_op(value: Operation | Block, ctx: InterpreterCtx): Attribute = {
    value match {
      case op: Operation => interpret_op(op, ctx)
      case block: Block => interpret_block(block, ctx)
    }
  }

  // TODO: implement block evaluation with envs
  def interpret_block(block: Block, ctx: InterpreterCtx): Attribute = {
    IntegerAttr(IntData(0), I32)
  }
}

package scair.tools

import scair.dialects.builtin.*
import scair.ir.*
import scair.dialects.arith

trait ArithmeticEvaluator:
  self: Interpreter =>

  def interpret_constant(constant: arith.Constant): Int =
    constant.value match
      case value: IntegerAttr =>
        value.value.toInt
      case _ =>
        throw new Exception("Unsupported value type for constant operation")

  def interpret_bin_op(
      lhs: Value[Attribute],
      rhs: Value[Attribute],
      ctx: InterpreterCtx
  )(combine: (Int, Int) => Int): Int = 
    val lval = lookup_op(lhs.owner.get, ctx)
    val rval = lookup_op(rhs.owner.get, ctx)
    (lval, rval) match
      case (lval: Int, rval: Int) =>
        combine(lval, rval)
      case _ =>
        throw new Exception("Unsupported operand types for binary operation")

  def interpret_cmp_op(
      lhs: Value[Attribute],
      rhs: Value[Attribute],
      predicate: Int,
      ctx: InterpreterCtx
  ): Int =
    val lval = lookup_op(lhs.owner.get, ctx)
    val rval = lookup_op(rhs.owner.get, ctx)
    (lval, rval) match
      case (lval: Int, rval: Int) =>
        predicate match
          case 0 => // EQ
            if lval == rval then 1 else 0
          case 1 => // NE
            if lval != rval then 1 else 0
          case 2 | 6 => // SLT and ULT, assume both numbers are already converted accoringly
            if lval < rval then 1 else 0
          case 3 | 7 => // SLE and ULE
            if lval <= rval then 1 else 0
          case 4 | 8 => // SGT and UGT
            if lval > rval then 1 else 0
          case 5 | 9 => // SGE and UGE
            if lval >= rval then 1 else 0
          case _ => throw new Exception("Unknown comparison predicate")
      case other =>
        throw new Exception("Unsupported operand types for cmpi")

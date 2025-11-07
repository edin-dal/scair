package scair.tools

import scair.dialects.arith
import scair.dialects.builtin.*
import scair.ir.*

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
      ctx: RunTimeCtx
  )(combine: (Int, Int) => Int): Int =
    val lval = lookup_op(lhs, ctx)
    val rval = lookup_op(rhs, ctx)
    (lval, rval) match
      case (lval: Int, rval: Int) =>
        combine(lval, rval)
      case _ =>
        throw new Exception("Unsupported operand types for binary operation")

  def interpret_cmp_op(
      lhs: Value[Attribute],
      rhs: Value[Attribute],
      predicate: Int,
      ctx: RunTimeCtx
  ): Boolean =
    val lval = lookup_op(lhs, ctx)
    val rval = lookup_op(rhs, ctx)
    (lval, rval) match
      case (lval: Int, rval: Int) =>
        predicate match
          case 0 => // EQ
            lval == rval
          case 1 => // NE
            lval != rval
          case 2 | 6 => // SLT and ULT, assume both numbers are already converted accoringly
            lval < rval
          case 3 | 7 => // SLE and ULE
            lval <= rval
          case 4 | 8 => // SGT and UGT
            lval > rval
          case 5 | 9 => // SGE and UGE
            lval >= rval
          case _ => throw new Exception("Unknown comparison predicate")
      case other =>
        throw new Exception("Unsupported operand types for cmpi")

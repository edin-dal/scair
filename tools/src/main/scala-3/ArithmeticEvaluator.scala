package scair.tools

import scair.ir.*
import scair.dialects.builtin.*

trait ArithmeticEvaluator { self: Interpreter =>

    def interpret_bin_op(lhs: Value[Attribute], rhs: Value[Attribute], ctx: InterpreterCtx)
    (combine: (Int, Int) => Int): Attribute = {
        val lval = lookup_op(lhs.owner.get, ctx)
        val rval = lookup_op(rhs.owner.get, ctx)
        (lval, rval) match {
        case (li: IntegerAttr, ri: IntegerAttr) =>
            IntegerAttr(IntData(combine(li.value.value.toInt, ri.value.value.toInt)), li.typ)
        case _ => throw new Exception(s"Unsupported operand types for binary operation")
        }
    }

    def interpret_cmp_op(lhs: Value[Attribute], rhs: Value[Attribute], predicate: Int, ctx: InterpreterCtx): Attribute = {
    val lval = lookup_op(lhs.owner.get, ctx)
    val rval = lookup_op(rhs.owner.get, ctx)
    (lval, rval) match {
      case (li: IntegerAttr, ri: IntegerAttr) =>
        val lcmp = li.value.value
        val rcmp = ri.value.value
        predicate match {
          case 0 => // EQ
            if (lcmp == rcmp) IntegerAttr(IntData(1), I1) else IntegerAttr(IntData(0), I1)
          case 1 => // NE
            if (lcmp != rcmp) IntegerAttr(IntData(1), I1) else IntegerAttr(IntData(0), I1)
          case 2 | 6 => // SLT and ULT, assume both numbers are already converted accoringly
            if (lcmp < rcmp) IntegerAttr(IntData(1), I1) else IntegerAttr(IntData(0), I1)
          case 3 | 7 => // SLE and ULE
            if (lcmp <= rcmp) IntegerAttr(IntData(1), I1) else IntegerAttr(IntData(0), I1)
          case 4 | 8 => // SGT and UGT
            if (lcmp > rcmp) IntegerAttr(IntData(1), I1) else IntegerAttr(IntData(0), I1)
          case 5 | 9 => // SGE and UGE
            if (lcmp >= rcmp) IntegerAttr(IntData(1), I1) else IntegerAttr(IntData(0), I1)
          case _ => throw new Exception("Unknown comparison predicate")
        }
      case other => throw new Exception("Unsupported operand types for cmpi, got: $other")
    }
  }
}
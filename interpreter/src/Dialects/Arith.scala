package scair.interpreter

import scair.dialects.arith
import scair.dialects.builtin.IntegerAttr
import scair.ir.*

given Conversion[Boolean, Int] = if _ then 1 else 0

object run_constant extends OpImpl[arith.Constant]:

  def run(op: arith.Constant, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
    op.value match
      case intAttr: IntegerAttr =>
        ctx.vars.put(op.result, intAttr.value.toInt)
      case _ => throw new Exception("Unsupported constant attribute type")

// may extend bin ops to BigInt later
object run_addi extends OpImpl[arith.AddI]:

  def run(op: arith.AddI, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
    val lhs = interpreter.lookup_op(op.lhs, ctx)
    val rhs = interpreter.lookup_op(op.rhs, ctx)
    ctx.vars.put(op.result, lhs + rhs)

object run_subi extends OpImpl[arith.SubI]:

  def run(op: arith.SubI, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
    val lhs = interpreter.lookup_op(op.lhs, ctx)
    val rhs = interpreter.lookup_op(op.rhs, ctx)
    ctx.vars.put(op.result, lhs - rhs)

object run_muli extends OpImpl[arith.MulI]:

  def run(op: arith.MulI, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
    val lhs = interpreter.lookup_op(op.lhs, ctx)
    val rhs = interpreter.lookup_op(op.rhs, ctx)
    ctx.vars.put(op.result, lhs * rhs)

object run_divsi extends OpImpl[arith.DivSI]:

  def run(op: arith.DivSI, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
    val lhs = interpreter.lookup_op(op.lhs, ctx)
    val rhs = interpreter.lookup_op(op.rhs, ctx)
    ctx.vars.put(op.result, lhs / rhs)

object run_divui extends OpImpl[arith.DivUI]:

  def run(op: arith.DivUI, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
    val lhs = interpreter.lookup_op(op.lhs, ctx)
    val rhs = interpreter.lookup_op(op.rhs, ctx)
    ctx.vars.put(op.result, lhs / rhs)

object run_andi extends OpImpl[arith.AndI]:

  def run(op: arith.AndI, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
    val lhs = interpreter.lookup_op(op.lhs, ctx)
    val rhs = interpreter.lookup_op(op.rhs, ctx)
    ctx.vars.put(op.result, lhs & rhs)

object run_ori extends OpImpl[arith.OrI]:

  def run(op: arith.OrI, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
    val lhs = interpreter.lookup_op(op.lhs, ctx)
    val rhs = interpreter.lookup_op(op.rhs, ctx)
    ctx.vars.put(op.result, lhs | rhs)

object run_xori extends OpImpl[arith.XOrI]:

  def run(op: arith.XOrI, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
    val lhs = interpreter.lookup_op(op.lhs, ctx)
    val rhs = interpreter.lookup_op(op.rhs, ctx)
    ctx.vars.put(op.result, lhs ^ rhs)

object run_shli extends OpImpl[arith.ShLI]:

  def run(op: arith.ShLI, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
    val lhs = interpreter.lookup_op(op.lhs, ctx)
    val rhs = interpreter.lookup_op(op.rhs, ctx)
    ctx.vars.put(op.result, lhs << rhs)

object run_shrsi extends OpImpl[arith.ShRSI]:

  def run(op: arith.ShRSI, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
    val lhs = interpreter.lookup_op(op.lhs, ctx)
    val rhs = interpreter.lookup_op(op.rhs, ctx)
    ctx.vars.put(op.result, lhs >> rhs)

object run_shrui extends OpImpl[arith.ShRUI]:

  def run(op: arith.ShRUI, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
    val lhs = interpreter.lookup_op(op.lhs, ctx)
    val rhs = interpreter.lookup_op(op.rhs, ctx)
    ctx.vars.put(op.result, lhs >>> rhs)

object run_cmpi extends OpImpl[arith.CmpI]:

  def run(op: arith.CmpI, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
    val lhs = interpreter.lookup_op(op.lhs, ctx)
    val rhs = interpreter.lookup_op(op.rhs, ctx)

    op.predicate.ordinal match
      case 0 => // EQ
        ctx.vars.put(op.result, (lhs == rhs): Int)
      case 1 => // NE
        ctx.vars.put(op.result, (lhs != rhs): Int)
      case 2 | 6 => // SLT and ULT
        ctx.vars.put(op.result, (lhs < rhs): Int)
      case 3 | 7 => // SLE and ULE
        ctx.vars.put(op.result, (lhs <= rhs): Int)
      case 4 | 8 => // SGT and UGT
        ctx.vars.put(op.result, (lhs > rhs): Int)
      case 5 | 9 => // SGE and UGE
        ctx.vars.put(op.result, (lhs >= rhs): Int)
      case _ => throw new Exception("Unknown comparison predicate")

object run_select extends OpImpl[arith.SelectOp]:

  def run(op: arith.SelectOp, interpreter: Interpreter, ctx: RuntimeCtx): Unit =
    interpreter.lookup_boollike(op.condition, ctx) match
      case 0 =>
        ctx.vars.put(
          op.result,
          interpreter.lookup_op(op.falseValue, ctx),
        )
      case 1 =>
        ctx.vars.put(
          op.result,
          interpreter.lookup_op(op.trueValue, ctx),
        )
      case _ => throw new Exception("Select condition must be 0 or 1")

val InterpreterArithDialect: InterpreterDialect =
  Seq(
    run_constant,
    run_addi,
    run_subi,
    run_muli,
    run_divsi,
    run_divui,
    run_andi,
    run_ori,
    run_xori,
    run_shli,
    run_shrsi,
    run_shrui,
    run_cmpi,
    run_select,
  )

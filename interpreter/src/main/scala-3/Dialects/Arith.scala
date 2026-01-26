package scair.interpreter

import scair.dialects.arith
import scair.dialects.builtin.IntegerAttr
import scair.ir.*

object run_constant extends OpImpl[arith.Constant]:

  def compute(op: arith.Constant, interpreter: Interpreter, ctx: RuntimeCtx): Any =
    op.value match
      case intAttr: IntegerAttr =>
        intAttr.value.toInt
      case _ => throw new Exception("Unsupported constant attribute type")

object run_addi extends OpImpl[arith.AddI]:

  def compute(op: arith.AddI, interpreter: Interpreter, ctx: RuntimeCtx): Any =

    val args = Seq(op.lhs, op.rhs)
    val operands = lookup_operands(args, interpreter, ctx)
    operands(0) + operands(1)

object run_subi extends OpImpl[arith.SubI]:

  def compute(op: arith.SubI, interpreter: Interpreter, ctx: RuntimeCtx): Any =
    val args = Seq(op.lhs, op.rhs)
    val operands = lookup_operands(args, interpreter, ctx)
    operands(0) - operands(1)

object run_muli extends OpImpl[arith.MulI]:

  def compute(op: arith.MulI, interpreter: Interpreter, ctx: RuntimeCtx): Any =
    val args = Seq(op.lhs, op.rhs)
    val operands = lookup_operands(args, interpreter, ctx)
    operands(0) * operands(1)

object run_divsi extends OpImpl[arith.DivSI]:

// TODO: signed division
  def compute(op: arith.DivSI, interpreter: Interpreter, ctx: RuntimeCtx): Any =
    val args = Seq(op.lhs, op.rhs)
    val operands = lookup_operands(args, interpreter, ctx)
    operands(0) / operands(1)

object run_divui extends OpImpl[arith.DivUI]:

  def compute(op: arith.DivUI, interpreter: Interpreter, ctx: RuntimeCtx): Any =
    val args = Seq(op.lhs, op.rhs)
    val operands = lookup_operands(args, interpreter, ctx)
    operands(0) / operands(1)

object run_andi extends OpImpl[arith.AndI]:

  def compute(op: arith.AndI, interpreter: Interpreter, ctx: RuntimeCtx): Any =
    val args = Seq(op.lhs, op.rhs)
    val operands = lookup_operands(args, interpreter, ctx)
    operands(0) & operands(1)

object run_ori extends OpImpl[arith.OrI]:

  def compute(op: arith.OrI, interpreter: Interpreter, ctx: RuntimeCtx): Any =
    val args = Seq(op.lhs, op.rhs)
    val operands = lookup_operands(args, interpreter, ctx)
    operands(0) | operands(1)

object run_xori extends OpImpl[arith.XOrI]:

  def compute(op: arith.XOrI, interpreter: Interpreter, ctx: RuntimeCtx): Any =
    val args = Seq(op.lhs, op.rhs)
    val operands = lookup_operands(args, interpreter, ctx)
    operands(0) ^ operands(1)

object run_shli extends OpImpl[arith.ShLI]:

  def compute(op: arith.ShLI, interpreter: Interpreter, ctx: RuntimeCtx): Any =
    val args = Seq(op.lhs, op.rhs)
    val operands = lookup_operands(args, interpreter, ctx)
    operands(0) << operands(1)

object run_shrsi extends OpImpl[arith.ShRSI]:

  def compute(op: arith.ShRSI, interpreter: Interpreter, ctx: RuntimeCtx): Any =
    val args = Seq(op.lhs, op.rhs)
    val operands = lookup_operands(args, interpreter, ctx)
    operands(0) >> operands(1)

object run_shrui extends OpImpl[arith.ShRUI]:

  def compute(op: arith.ShRUI, interpreter: Interpreter, ctx: RuntimeCtx): Any =
    val args = Seq(op.lhs, op.rhs)
    val operands = lookup_operands(args, interpreter, ctx)
    operands(0) >>> operands(1)

object run_cmpi extends OpImpl[arith.CmpI]:

  def compute(op: arith.CmpI, interpreter: Interpreter, ctx: RuntimeCtx): Any =
    val args = Seq(op.lhs, op.rhs)
    val operands = lookup_operands(args, interpreter, ctx)

    op.predicate.ordinal match
      case 0 => // EQ
        ctx.scopedDict.update(op.result, (operands(0) == operands(1)))
      case 1 => // NE
        ctx.scopedDict.update(op.result, (operands(0) != operands(1)))
      case 2 | 6 => // SLT and ULT
        ctx.scopedDict.update(op.result, (operands(0) < operands(1)))
      case 3 | 7 => // SLE and ULE
        ctx.scopedDict.update(op.result, (operands(0) <= operands(1)))
      case 4 | 8 => // SGT and UGT
        ctx.scopedDict.update(op.result, (operands(0) > operands(1)))
      case 5 | 9 => // SGE and UGE
        ctx.scopedDict.update(op.result, (operands(0) >= operands(1)))
      case _ => throw new Exception("Unknown comparison predicate")

object run_select extends OpImpl[arith.SelectOp]:

  def compute(op: arith.SelectOp, interpreter: Interpreter, ctx: RuntimeCtx): Any =
    val cond = Seq(op.condition)
    
    interpreter.lookup_boollike(op.condition, ctx) match
      case 0 =>
        ctx.scopedDict.update(
          op.result,
          interpreter.lookup_op(op.falseValue, ctx),
        )
      case 1 =>
        ctx.scopedDict.update(
          op.result,
          interpreter.lookup_op(op.trueValue, ctx),
        )
      case _ => throw new Exception("Select condition must be 0 or 1")

val InterpreterArithDialect: InterpreterDialect =
  Seq(
    run_constant,
    // run_subi,
    // run_muli,
    // run_divsi,
    // run_divui,
    // run_andi,
    // run_ori,
    // run_xori,
    // run_shli,
    // run_shrsi,
    // run_shrui,
    // run_cmpi,
    // run_select,
    run_addi,
  )

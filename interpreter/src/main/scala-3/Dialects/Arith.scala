package scair.tools

import scair.dialects.arith
import scair.dialects.builtin.*
import scair.ir.*

// implicit helper function to convert Boolean to Int
// is this necessary?
class asInt(b: Boolean) {
  def toInt = if(b) 1 else 0
}

implicit def convertBooleanToInt(b: Boolean): asInt = new asInt(b)

def run_constant(op: arith.Constant, ctx: RuntimeCtx): Unit =
  op.value match
    case value: IntegerAttr =>
      ctx.vars.put(op.result, value.value.toInt)
    case _ =>
      throw new Exception("Unsupported value type for constant operation")

// may extend bin ops to BigInt later
def run_addi(op: arith.AddI, ctx: RuntimeCtx): Unit = 
  val lhs = lookup_op(op.lhs, ctx)
  val rhs = lookup_op(op.rhs, ctx)
  ctx.vars.put(op.result, lhs + rhs)

def run_subi(op: arith.SubI, ctx: RuntimeCtx): Unit = 
  val lhs = lookup_op(op.lhs, ctx)
  val rhs = lookup_op(op.rhs, ctx)
  ctx.vars.put(op.result, lhs - rhs)

def run_muli(op: arith.MulI, ctx: RuntimeCtx): Unit = 
  val lhs = lookup_op(op.lhs, ctx)
  val rhs = lookup_op(op.rhs, ctx)
  ctx.vars.put(op.result, lhs * rhs)

def run_divsi(op: arith.DivSI, ctx: RuntimeCtx): Unit = 
  val lhs = lookup_op(op.lhs, ctx)
  val rhs = lookup_op(op.rhs, ctx)
  ctx.vars.put(op.result, lhs / rhs)

def run_divui(op: arith.DivUI, ctx: RuntimeCtx): Unit = 
  val lhs = lookup_op(op.lhs, ctx)
  val rhs = lookup_op(op.rhs, ctx)
  ctx.vars.put(op.result, lhs / rhs)

def run_andi(op: arith.AndI, ctx: RuntimeCtx): Unit = 
  val lhs = lookup_op(op.lhs, ctx)
  val rhs = lookup_op(op.rhs, ctx)
  ctx.vars.put(op.result, lhs & rhs)

def run_ori(op: arith.OrI, ctx: RuntimeCtx): Unit = 
  val lhs = lookup_op(op.lhs, ctx)
  val rhs = lookup_op(op.rhs, ctx)
  ctx.vars.put(op.result, lhs | rhs)

def run_xori(op: arith.XOrI, ctx: RuntimeCtx): Unit = 
  val lhs = lookup_op(op.lhs, ctx)
  val rhs = lookup_op(op.rhs, ctx)
  ctx.vars.put(op.result, lhs ^ rhs)

def run_shli(op: arith.ShLI, ctx: RuntimeCtx): Unit = 
  val lhs = lookup_op(op.lhs, ctx)
  val rhs = lookup_op(op.rhs, ctx)
  ctx.vars.put(op.result, lhs << rhs)

def run_shrsi(op: arith.ShRSI, ctx: RuntimeCtx): Unit = 
  val lhs = lookup_op(op.lhs, ctx)
  val rhs = lookup_op(op.rhs, ctx)
  ctx.vars.put(op.result, lhs >> rhs)

def run_shrui(op: arith.ShRUI, ctx: RuntimeCtx): Unit = 
  val lhs = lookup_op(op.lhs, ctx)
  val rhs = lookup_op(op.rhs, ctx)
  ctx.vars.put(op.result, lhs >>> rhs)

def run_cmpi(op: arith.CmpI, ctx: RuntimeCtx): Unit = 
  val lhs = lookup_op(op.lhs, ctx)
  val rhs = lookup_op(op.rhs, ctx)

  op.predicate.value.toInt match
    case 0 => // EQ
      ctx.vars.put(op.result, (lhs == rhs).toInt)
    case 1 => // NE
      ctx.vars.put(op.result, (lhs != rhs).toInt)
    case 2 | 6 => // SLT and ULT
      ctx.vars.put(op.result, (lhs < rhs).toInt)
    case 3 | 7 => // SLE and ULE
      ctx.vars.put(op.result, (lhs <= rhs).toInt)
    case 4 | 8 => // SGT and UGT
      ctx.vars.put(op.result, (lhs > rhs).toInt)
    case 5 | 9 => // SGE and UGE
      ctx.vars.put(op.result, (lhs >= rhs).toInt)
    case _ => throw new Exception("Unknown comparison predicate")
  

def run_select(op: arith.SelectOp, ctx: RuntimeCtx): Unit = 
  lookup_boollike(op.condition, ctx) match
    case 0 =>
      ctx.vars.put(
        op.result,
        lookup_op(op.falseValue, ctx)
      )
    case 1 =>
      ctx.vars.put(
        op.result,
        lookup_op(op.trueValue, ctx)
      )
    case _ => throw new Exception("Select condition must be 0 or 1")

val InterpreterArithDialect = summonImplementations(
  Seq(
    OpImpl(run_constant),
    OpImpl(run_addi),
    OpImpl(run_subi),
    OpImpl(run_muli),
    OpImpl(run_divsi),
    OpImpl(run_divui),
    OpImpl(run_andi),
    OpImpl(run_ori),
    OpImpl(run_xori),
    OpImpl(run_shli),
    OpImpl(run_shrsi),
    OpImpl(run_shrui),
    OpImpl(run_cmpi),
    OpImpl(run_select)
  )
)

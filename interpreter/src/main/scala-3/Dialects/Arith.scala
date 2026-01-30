package scair.interpreter

import scair.dialects.arith
import scair.dialects.builtin.IntegerAttr
import scair.ir.*
import scair.dialects.arith.CmpIPredicate

object run_constant extends OpImpl[arith.Constant]:

  def compute(
      op: arith.Constant,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Tuple,
  ): Any =
    op.value match
      case intAttr: IntegerAttr =>
        intAttr.value.toInt
      case _ => throw new Exception("Unsupported constant attribute type")

object run_addi extends OpImpl[arith.AddI]:

  def compute(
      op: arith.AddI,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Tuple,
  ): Any =
    args match
      case (lhs: Int, rhs: Int) =>
        lhs + rhs
      case _ => throw new Exception("AddI operands must be integers")

object run_subi extends OpImpl[arith.SubI]:

  def compute(
      op: arith.SubI,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Tuple,
  ): Any =
    args match
      case (lhs: Int, rhs: Int) =>
        lhs - rhs
      case _ => throw new Exception("SubI operands must be integers")

object run_muli extends OpImpl[arith.MulI]:

  def compute(
      op: arith.MulI,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Tuple,
  ): Any =
    args match
      case (lhs: Int, rhs: Int) =>
        lhs * rhs
      case _ => throw new Exception("MulI operands must be integers")

object run_divsi extends OpImpl[arith.DivSI]:

// TODO: signed division
  def compute(
      op: arith.DivSI,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Tuple,
  ): Any =
    args match
      case (lhs: Int, rhs: Int) =>
        lhs / rhs
      case _ => throw new Exception("DivSI operands must be integers")

object run_divui extends OpImpl[arith.DivUI]:

  def compute(
      op: arith.DivUI,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Tuple,
  ): Any =
    args match
      case (lhs: Int, rhs: Int) =>
        lhs / rhs
      case _ => throw new Exception("DivUI operands must be integers")

object run_andi extends OpImpl[arith.AndI]:

  def compute(
      op: arith.AndI,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Tuple,
  ): Any =
    args match
      case (lhs: Int, rhs: Int) =>
        lhs & rhs
      case _ => throw new Exception("AndI operands must be integers")

object run_ori extends OpImpl[arith.OrI]:

  def compute(
      op: arith.OrI,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Tuple,
  ): Any =
    args match
      case (lhs: Int, rhs: Int) =>
        lhs | rhs
      case _ => throw new Exception("OrI operands must be integers")

object run_xori extends OpImpl[arith.XOrI]:

  def compute(
      op: arith.XOrI,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Tuple,
  ): Any =
    args match
      case (lhs: Int, rhs: Int) =>
        lhs ^ rhs
      case _ => throw new Exception("XOrI operands must be integers")

object run_shli extends OpImpl[arith.ShLI]:

  def compute(
      op: arith.ShLI,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Tuple,
  ): Any =
    args match
      case (lhs: Int, rhs: Int) =>
        lhs << rhs
      case _ => throw new Exception("ShLI operands must be integers")

object run_shrsi extends OpImpl[arith.ShRSI]:

  def compute(
      op: arith.ShRSI,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Tuple,
  ): Any =
    args match
      case (lhs: Int, rhs: Int) =>
        lhs >> rhs
      case _ => throw new Exception("ShRSI operands must be integers")

object run_shrui extends OpImpl[arith.ShRUI]:

  def compute(
      op: arith.ShRUI,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Tuple,
  ): Any =
    args match
      case (lhs: Int, rhs: Int) =>
        lhs >>> rhs
      case _ => throw new Exception("ShRUI operands must be integers")


// TODO: signedness
object run_cmpi extends OpImpl[arith.CmpI]:

  def compute(
      op: arith.CmpI,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Tuple,
  ): Any =
    args match
      case (lhs: Int, rhs: Int) =>
        op.predicate match
          case CmpIPredicate.eq =>
            (lhs == rhs)
          case CmpIPredicate.ne => 
            (lhs != rhs)
          case CmpIPredicate.slt | CmpIPredicate.ult =>
            (lhs < rhs)
          case CmpIPredicate.sle | CmpIPredicate.ule =>
            (lhs <= rhs)
          case CmpIPredicate.sgt | CmpIPredicate.ugt =>
            (lhs > rhs)
          case CmpIPredicate.sge | CmpIPredicate.uge =>
            (lhs >= rhs)
          case _ => throw new Exception("Unknown comparison predicate")
      case _ => throw new Exception("CmpI operands must be integers")

object run_select extends OpImpl[arith.SelectOp]:

  def compute(
      op: arith.SelectOp,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Tuple,
  ): Any =
    args match
      case (cond, trueVal, falseVal) =>
        if cond == 1 then trueVal else falseVal
      case _ =>
        throw new Exception("Select operands must be (Boolean, Int, Int)")

val InterpreterArithDialect: InterpreterDialect =
  Seq(
    run_constant,
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
    run_addi,
  )

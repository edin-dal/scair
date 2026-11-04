package scair.interpreter

import scair.dialects.llvm

object run_llvm_br extends OpTerminatorImpl[llvm.Br]:

  def computeTerminator(
      op: llvm.Br,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Seq[Any],
  ): (CFGStep, Seq[Any]) =
    (CFGStep.Jump(op.dest, args), Seq())

object run_llvm_cond_br extends OpTerminatorImpl[llvm.CondBr]:

  def computeTerminator(
      op: llvm.CondBr,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Seq[Any],
  ): (CFGStep, Seq[Any]) =
    args match
      case cond +: rest =>
        val (trueArgs, falseArgs) = rest.splitAt(op.trueArgs.size)
        val taken = cond match
          case true => true
          case 1    => true
          case _    => false
        val step =
          if taken then CFGStep.Jump(op.trueDest, trueArgs)
          else CFGStep.Jump(op.falseDest, falseArgs)
        (step, Seq())
      case _ =>
        throw new Exception("llvm.cond_br requires a condition operand")

object run_llvm_return extends OpTerminatorImpl[llvm.Return]:

  def computeTerminator(
      op: llvm.Return,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Seq[Any],
  ): (CFGStep, Seq[Any]) =
    (CFGStep.Return(args), Seq())

object run_llvm_unreachable extends OpTerminatorImpl[llvm.Unreachable]:

  def computeTerminator(
      op: llvm.Unreachable,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Seq[Any],
  ): (CFGStep, Seq[Any]) =
    throw new Exception("llvm.unreachable executed")

val InterpreterLLVMDialect: InterpreterDialect =
  Seq(run_llvm_br, run_llvm_cond_br, run_llvm_return, run_llvm_unreachable)

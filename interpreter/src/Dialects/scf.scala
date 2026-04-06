package scair.interpreter

import scair.dialects.scf

object run_for extends OpImpl[scf.ForOp]:

  def compute(
      op: scf.ForOp,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Seq[Any],
  ): Seq[Any] =
    args match
        case Seq(lowerBound: Int, upperBound: Int, step: Int, rest @ _*) =>
            var loopArgs: Seq[Any] = rest.toSeq

            val blockArgs = op.region.blocks.head.arguments

            for i <- lowerBound until upperBound by step do
                val iterCtx = ctx.push_scope(s"for_loop_iter_$i")

                // Bind induction variable to region entry arg 0
                iterCtx.scopedDict.update(blockArgs(0), i)

                // Bind loop-carried args to region entry args 1..n
                loopArgs.zipWithIndex.foreach { case (value, idx) =>
                iterCtx.scopedDict.update(blockArgs(idx + 1), value)
                }
                loopArgs = i +: loopArgs // Prepend induction variable to loop args for next iteration

                // Execute loop body. Expect scf.yield values as region result.
                val yielded = interpreter.run_ssacfg_region(
                op.region,
                iterCtx,
                s"for_loop_iter_$i",
                loopArgs
                )
                loopArgs = yielded
            loopArgs
        case _ =>
            throw new Exception("For loop operands must be (Int, Int, Int, ...iter_args)")

object run_if extends OpImpl[scf.IfOp]:

  def compute(
      op: scf.IfOp,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Seq[Any],
  ): Seq[Any] =
    args match
        case Seq(cond: Int, rest @ _*) =>
            val regionToRun = if cond == 1 then op.thenRegion else op.elseRegion
            interpreter.run_ssacfg_region(regionToRun, ctx.push_scope("if_branch"), "if_branch", rest.toSeq)
        case Seq(cond: Boolean, rest @ _*) =>
            val regionToRun = if cond then op.thenRegion else op.elseRegion
            interpreter.run_ssacfg_region(regionToRun, ctx.push_scope("if_branch"), "if_branch", rest.toSeq)
        case _ =>
            throw new Exception("If condition must be an integer")

object run_yield extends OpImpl[scf.YieldOp]:

  def compute(
      op: scf.YieldOp,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Seq[Any],
  ): Seq[Any] =
    args

val InterpreterScfDialect: InterpreterDialect =
  Seq(
    run_for,
    run_yield,
    run_if,
  )

package scair.passes.convert_func_to_llvm

import scair.MLContext
import scair.dialects.func
import scair.dialects.llvm
import scair.ir.*
import scair.transformations.GreedyRewritePatternApplier
import scair.transformations.PatternRewriteWalker
import scair.transformations.WalkerPass
import scair.transformations.pattern

private val LowerFunc = pattern { case op: func.Func =>
  val lowered = llvm.Func(
    op.sym_name,
    op.function_type,
    if op.body.blocks.isEmpty then None else op.sym_visibility,
    op.body.detached,
  )
  lowered.attributes.addAll(op.attributes)
  lowered
}

private val LowerCall = pattern { case call: func.Call =>
  llvm.Call(
    call.callee,
    call._operands,
    call._results.map(_.copy()),
  )
}

private val LowerReturn = pattern { case ret: func.Return =>
  llvm.Return(ret._operands)
}

final class ConvertFuncToLLVM(ctx: MLContext) extends WalkerPass(ctx):
  override val name: String = "convert-func-to-llvm"

  override val walker: PatternRewriteWalker =
    PatternRewriteWalker(
      GreedyRewritePatternApplier(Seq(LowerFunc, LowerCall, LowerReturn))
    )

package scair.transformations.benchmark_constant_folding

import scair.dialects.arith.AddI
import scair.dialects.arith.Constant
import scair.dialects.builtin.IntegerAttr
import scair.ir.Operation
import scair.ir.Result
import scair.transformations.GreedyRewritePatternApplier
import scair.transformations.ModulePass
import scair.transformations.PatternRewriteWalker
import scair.transformations.canonicalization.RemoveUnusedOperations
import scair.transformations.patterns.Owner
import scair.transformations.patterns.pattern

val AddIfold = pattern {
  case AddI(
        Owner(Constant(c0: IntegerAttr, _)),
        Owner(Constant(c1: IntegerAttr, _)),
        _
      ) =>
    Constant(c0 + c1, Result(c0.typ))
}

object BenchmarkConstantFolding extends ModulePass {
  override val name = "benchmark-constant-folding"

  override def transform(op: Operation): Operation = {
    val prw = new PatternRewriteWalker(
      GreedyRewritePatternApplier(Seq(AddIfold, RemoveUnusedOperations))
    )
    prw.rewrite_op(op)
    return op
  }

}

package scair.transformations.benchmark_constant_folding

import scair.ir.{Operation, Result}
import scair.transformations.patterns.{pattern, Owner}
import scair.dialects.builtin.IntegerAttr
import scair.dialects.arith.{AddI, Constant}
import scair.transformations.{ModulePass, PatternRewriteWalker, GreedyRewritePatternApplier}
import scair.transformations.canonicalization.RemoveUnusedOperations

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
    val prw = new PatternRewriteWalker(GreedyRewritePatternApplier(Seq(AddIfold, RemoveUnusedOperations)))
    prw.rewrite_op(op)
    return op
  }

}

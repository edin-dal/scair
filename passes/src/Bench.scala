package scair.passes.benchmark_constant_folding

import scair.MLContext
import scair.dialects.arith.AddI
import scair.dialects.arith.Constant
import scair.dialects.builtin.IntegerAttr
import scair.ir.Result
import scair.passes.canonicalization.RemoveUnusedOperations
import scair.transformations.GreedyRewritePatternApplier
import scair.transformations.PatternRewriteWalker
import scair.transformations.WalkerPass
import scair.transformations.patterns.Owner
import scair.transformations.patterns.pattern

//
// ██████╗░ ███████╗ ███╗░░██╗ ░█████╗░ ██╗░░██╗
// ██╔══██╗ ██╔════╝ ████╗░██║ ██╔══██╗ ██║░░██║
// ██████╦╝ █████╗░░ ██╔██╗██║ ██║░░╚═╝ ███████║
// ██╔══██╗ ██╔══╝░░ ██║╚████║ ██║░░██╗ ██╔══██║
// ██████╦╝ ███████╗ ██║░╚███║ ╚█████╔╝ ██║░░██║
// ╚═════╝░ ╚══════╝ ╚═╝░░╚══╝ ░╚════╝░ ╚═╝░░╚═╝
//

val AddIfold = pattern {
  case AddI(
        Owner(Constant(c0: IntegerAttr, _)),
        Owner(Constant(c1: IntegerAttr, _)),
        _,
      ) =>
    Constant(c0 + c1, Result(c0.typ))
}

final class BenchmarkConstantFolding(ctx: MLContext) extends WalkerPass(ctx):
  override val name = "benchmark-constant-folding"

  override final val walker = PatternRewriteWalker(
    GreedyRewritePatternApplier(Seq(AddIfold, RemoveUnusedOperations))
  )

package scair.passes.benchmark_constant_folding

import scair.MLContext
import scair.dialects.arith.AddI
import scair.dialects.arith.Constant
import scair.dialects.builtin.IntegerAttr
import scair.ir.Result
import scair.passes.canonicalization.RemoveUnusedOperations
import scair.transformations.GreedyRewritePatternApplier
import scair.transformations.Owner
import scair.transformations.PatternRewriteWalker
import scair.transformations.WalkerPass
import scair.transformations.pattern

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
        lhs = Owner(Constant(c0: IntegerAttr, _)),
        rhs = Owner(Constant(c1: IntegerAttr, _)),
      ) =>
    Constant(c0 + c1, Result(c0.typ))
}

final class BenchmarkConstantFolding(ctx: MLContext) extends WalkerPass(ctx):
  override val name = "benchmark-constant-folding"

  override final val walker = PatternRewriteWalker(
    GreedyRewritePatternApplier(Seq(AddIfold, RemoveUnusedOperations))
  )

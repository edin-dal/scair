package scair.passes.canonicalization

import scair.MLContext
import scair.ir.*
import scair.transformations.*

// ░██████╗░ ███████╗ ███╗░░██╗ ███████╗ ██████╗░ ██╗ ░█████╗░
// ██╔════╝░ ██╔════╝ ████╗░██║ ██╔════╝ ██╔══██╗ ██║ ██╔══██╗
// ██║░░██╗░ █████╗░░ ██╔██╗██║ █████╗░░ ██████╔╝ ██║ ██║░░╚═╝
// ██║░░╚██╗ ██╔══╝░░ ██║╚████║ ██╔══╝░░ ██╔══██╗ ██║ ██║░░██╗
// ╚██████╔╝ ███████╗ ██║░╚███║ ███████╗ ██║░░██║ ██║ ╚█████╔╝
// ░╚═════╝░ ╚══════╝ ╚═╝░░╚══╝ ╚══════╝ ╚═╝░░╚═╝ ╚═╝ ░╚════╝░
//
// ░█████╗░ ░█████╗░ ███╗░░██╗ ░█████╗░ ███╗░░██╗ ██╗ ░█████╗░ ░█████╗░ ██╗░░░░░ ██╗ ███████╗ ░█████╗░ ████████╗ ██╗ ░█████╗░ ███╗░░██╗
// ██╔══██╗ ██╔══██╗ ████╗░██║ ██╔══██╗ ████╗░██║ ██║ ██╔══██╗ ██╔══██╗ ██║░░░░░ ██║ ╚════██║ ██╔══██╗ ╚══██╔══╝ ██║ ██╔══██╗ ████╗░██║
// ██║░░╚═╝ ███████║ ██╔██╗██║ ██║░░██║ ██╔██╗██║ ██║ ██║░░╚═╝ ███████║ ██║░░░░░ ██║ ░░███╔═╝ ███████║ ░░░██║░░░ ██║ ██║░░██║ ██╔██╗██║
// ██║░░██╗ ██╔══██║ ██║╚████║ ██║░░██║ ██║╚████║ ██║ ██║░░██╗ ██╔══██║ ██║░░░░░ ██║ ██╔══╝░░ ██╔══██║ ░░░██║░░░ ██║ ██║░░██║ ██║╚████║
// ╚█████╔╝ ██║░░██║ ██║░╚███║ ╚█████╔╝ ██║░╚███║ ██║ ╚█████╔╝ ██║░░██║ ███████╗ ██║ ███████╗ ██║░░██║ ░░░██║░░░ ██║ ╚█████╔╝ ██║░╚███║
// ░╚════╝░ ╚═╝░░╚═╝ ╚═╝░░╚══╝ ░╚════╝░ ╚═╝░░╚══╝ ╚═╝ ░╚════╝░ ╚═╝░░╚═╝ ╚══════╝ ╚═╝ ╚══════╝ ╚═╝░░╚═╝ ░░░╚═╝░░░ ╚═╝ ░╚════╝░ ╚═╝░░╚══╝
//

// TODO: Move out
// TODO: This is narrower than MLIR's `wouldOpBeTriviallyDead`, which also spares
//       symbols and allows erasing operations that only read or allocate.
val RemoveUnusedOperations = pattern {
  case _: IsTerminator => PatternAction.Abort
  case op: Operation
      if isMemoryEffectFree(op) && op.results.forall(_.uses.isEmpty) =>
    PatternAction.Erase
}

// TODO: Move out
val Commute = pattern { case c: Commutative =>
  val (const, nconst) = c.operands.partition(_.owner match
    case Some(_: ConstantLike) => true
    case _                     => false)
  val nops = nconst ++ const
  if nops == c.operands then PatternAction.Abort
  else c.updated(operands = nops)
}

final class Canonicalize(ctx: MLContext) extends WalkerPass(ctx):
  override val name = "canonicalize"

  lazy val canonicalizationPatterns = ctx.dialectOpContext.valuesIterator
    .flatMap(_.canonicalizationPatterns).toSeq

  override final val walker = PatternRewriteWalker(
    GreedyRewritePatternApplier(
      Seq(
        RemoveUnusedOperations,
        Commute,
      ) ++ canonicalizationPatterns
    )
  )

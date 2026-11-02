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

final case class Canonicalize() extends WalkerPass derives PassCompanion:

  override def walker(using ctx: MLContext) =
    val canonicalizationPatterns = ctx.dialectOpContext.valuesIterator
      .flatMap(_.canonicalizationPatterns).toSeq
    PatternRewriteWalker(
      GreedyRewritePatternApplier(
        Seq(
          RemoveUnusedOperations,
          Commute,
        ) ++ canonicalizationPatterns
      )
    )

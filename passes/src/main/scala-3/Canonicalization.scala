package scair.passes.canonicalization

import scair.MLContext
import scair.ir.*
import scair.transformations.*
import scair.transformations.patterns.*

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
val RemoveUnusedOperations = pattern {
  case _: IsTerminator => PatternAction.Abort
  case op: NoMemoryEffect if op.results.forall(_.uses.isEmpty) =>
    PatternAction.Erase
  case op: NoMemoryEffect => PatternAction.Abort
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
    .flatMap(_.canonicalizationPatterns)
    .toSeq

  override final val walker = PatternRewriteWalker(
    GreedyRewritePatternApplier(
      Seq(
        RemoveUnusedOperations,
        Commute
      ) ++ canonicalizationPatterns
    )
  )

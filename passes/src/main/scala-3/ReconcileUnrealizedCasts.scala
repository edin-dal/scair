package scair.passes.reconcile

import scair.MLContext
import scair.dialects.builtin.*
import scair.ir.*
import scair.transformations.*
import scair.transformations.patterns.*

import scala.annotation.tailrec

//
// ██████╗░ ███████╗ ░█████╗░ ░█████╗░ ███╗░░██╗ ░█████╗░ ██╗ ██╗░░░░░ ███████╗
// ██╔══██╗ ██╔════╝ ██╔══██╗ ██╔══██╗ ████╗░██║ ██╔══██╗ ██║ ██║░░░░░ ██╔════╝
// ██████╔╝ █████╗░░ ██║░░╚═╝ ██║░░██║ ██╔██╗██║ ██║░░╚═╝ ██║ ██║░░░░░ █████╗░░
// ██╔══██╗ ██╔══╝░░ ██║░░██╗ ██║░░██║ ██║╚████║ ██║░░██╗ ██║ ██║░░░░░ ██╔══╝░░
// ██║░░██║ ███████╗ ╚█████╔╝ ╚█████╔╝ ██║░╚███║ ╚█████╔╝ ██║ ███████╗ ███████╗
// ╚═╝░░╚═╝ ╚══════╝ ░╚════╝░ ░╚════╝░ ╚═╝░░╚══╝ ░╚════╝░ ╚═╝ ╚══════╝ ╚══════╝
//
// ██╗░░░██╗ ███╗░░██╗ ██████╗░ ███████╗ ░█████╗░ ██╗░░░░░ ██╗ ███████╗ ███████╗ ██████╗░
// ██║░░░██║ ████╗░██║ ██╔══██╗ ██╔════╝ ██╔══██╗ ██║░░░░░ ██║ ╚════██║ ██╔════╝ ██╔══██╗
// ██║░░░██║ ██╔██╗██║ ██████╔╝ █████╗░░ ███████║ ██║░░░░░ ██║ ░░███╔═╝ █████╗░░ ██║░░██║
// ██║░░░██║ ██║╚████║ ██╔══██╗ ██╔══╝░░ ██╔══██║ ██║░░░░░ ██║ ██╔══╝░░ ██╔══╝░░ ██║░░██║
// ╚██████╔╝ ██║░╚███║ ██║░░██║ ███████╗ ██║░░██║ ███████╗ ██║ ███████╗ ███████╗ ██████╔╝
// ░╚═════╝░ ╚═╝░░╚══╝ ╚═╝░░╚═╝ ╚══════╝ ╚═╝░░╚═╝ ╚══════╝ ╚═╝ ╚══════╝ ╚══════╝ ╚═════╝░
//
// ░█████╗░ ░█████╗░ ░██████╗ ████████╗ ░██████╗
// ██╔══██╗ ██╔══██╗ ██╔════╝ ╚══██╔══╝ ██╔════╝
// ██║░░╚═╝ ███████║ ╚█████╗░ ░░░██║░░░ ╚█████╗░
// ██║░░██╗ ██╔══██║ ░╚═══██╗ ░░░██║░░░ ░╚═══██╗
// ╚█████╔╝ ██║░░██║ ██████╔╝ ░░░██║░░░ ██████╔╝
// ░╚════╝░ ╚═╝░░╚═╝ ╚═════╝░ ░░░╚═╝░░░ ╚═════╝░
//

val SameType = pattern {
  case UnrealizedConversionCastOp(
        inputs = operands,
        outputs = results,
      ) if operands.typ == results.typ =>
    (Seq(), operands)
}

val Unused = pattern {
  case UnrealizedConversionCastOp(outputs = o) if o.forall(_.uses.isEmpty) =>
    PatternAction.Erase
}

@tailrec
def findCycleRootRec(
    cast: UnrealizedConversionCastOp,
    target: Seq[Attribute],
): Option[UnrealizedConversionCastOp] = cast match
  case UnrealizedConversionCastOp(Seq(), _) => None
  case UnrealizedConversionCastOp(i, o)     =>
    i.head.owner match
      case Some(parent @ UnrealizedConversionCastOp(inputs, outputs))
          if outputs == cast.inputs =>
        inputs.typ match
          case t if t == target => Some(parent)
          case _                => findCycleRootRec(parent, target)
      case _ => None

def findCycleRoot(
    cast: UnrealizedConversionCastOp
): Option[UnrealizedConversionCastOp] =
  findCycleRootRec(cast, cast.outputs.typ)

val InputFuse = pattern { case matched: UnrealizedConversionCastOp =>
  findCycleRoot(matched) match
    case Some(root) if root != matched && matched.containerBlock.isDefined =>
      UnrealizedConversionCastOp(
        inputs = root.inputs,
        outputs = matched.outputs,
      )
    case _ => PatternAction.Abort
}

final class ReconcileUnrealizedCasts(ctx: MLContext) extends WalkerPass(ctx):
  override val name = "reconcile-unrealized-casts"

  override final val walker = PatternRewriteWalker(
    GreedyRewritePatternApplier(Seq(SameType, Unused, InputFuse))
  )

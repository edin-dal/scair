package scair.transformations

import scair.dialects.builtin.*
import scair.ir.*

import scala.annotation.tailrec

//
// ██████╗░ ███████╗ ░█████╗░ ░█████╗░ ███╗░░██╗ ░█████╗░ ██╗ ██╗░░░░░ ███████╗
// ██╔══██╗ ██╔════╝ ██╔══██╗ ██╔══██╗ ████╗░██║ ██╔══██╗ ██║ ██║░░░░░ ██╔════╝
// ██████╔╝ █████╗░░ ██║░░╚═╝ ██║░░██║ ██╔██╗██║ ██║░░╚═╝ ██║ ██║░░░░░ █████╗░░
// ██╔══██╗ ██╔══╝░░ ██║░░██╗ ██║░░██║ ██║╚████║ ██║░░██╗ ██║ ██║░░░░░ ██╔══╝░░
// ██║░░██║ ███████╗ ╚█████╔╝ ╚█████╔╝ ██║░╚███║ ╚█████╔╝ ██║ ███████╗ ███████╗
// ╚═╝░░╚═╝ ╚══════╝ ░╚════╝░ ░╚════╝░ ╚═╝░░╚══╝ ░╚════╝░ ╚═╝ ╚══════╝ ╚══════╝
//

/** Folding of `builtin.unrealized_conversion_cast`.
  *
  * Lives in core so that a [[ConversionPass]] can reconcile the casts it
  * materialized; `reconcile-unrealized-casts` wraps the same patterns.
  */
object ReconcileCasts:

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

  val patterns: Seq[RewritePattern] = Seq(SameType, Unused, InputFuse)

  // A fresh walker per use: PatternRewriteWalker carries a worklist.
  def walker: PatternRewriteWalker =
    PatternRewriteWalker(GreedyRewritePatternApplier(patterns))

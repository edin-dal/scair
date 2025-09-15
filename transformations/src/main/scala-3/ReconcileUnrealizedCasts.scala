package scair.transformations.reconcile

import scair.dialects.builtin.*
import scair.ir.*
import scair.transformations.*
import scair.transformations.patterns.*

import scala.annotation.tailrec

val SameType = pattern {
  case UnrealizedConversionCastOp(
        inputs = operands,
        outputs = results
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
    target: Seq[Attribute]
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
    case Some(root) if root != matched && matched.container_block.isDefined =>
      UnrealizedConversionCastOp(
        inputs = root.inputs,
        outputs = matched.outputs
      )
    case _ => PatternAction.Abort
}

object ReconcileUnrealizedCasts extends ModulePass {
  override val name = "reconcile-unrealized-casts"

  override def transform(op: Operation): Operation = {
    val prw = new PatternRewriteWalker(
      GreedyRewritePatternApplier(Seq(SameType, Unused, InputFuse))
    )
    prw.rewrite_op(op)

    return op
  }

}

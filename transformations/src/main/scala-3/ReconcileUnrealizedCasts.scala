package scair.transformations.reconcile

import scair.dialects.builtin.*
import scair.ir.*
import scair.transformations.*
import scair.transformations.patterns.*

val SameType = pattern {
  case UnrealizedConversionCastOp(
        inputs = Seq(operand),
        outputs = Seq(result)
      ) if operand.typ == result.typ =>
    (Seq(), Seq(operand))
}

val Unused = pattern {
  case UnrealizedConversionCastOp(outputs = o) if o.forall(_.uses.isEmpty) =>
    PatternAction.Erase
}

object ReconcileUnrealizedCasts extends ModulePass {
  override val name = "reconcile-unrealized-casts"

  override def transform(op: Operation): Operation = {
    val prw = new PatternRewriteWalker(
      GreedyRewritePatternApplier(Seq(SameType, Unused))
    )
    prw.rewrite_op(op)

    return op
  }

}

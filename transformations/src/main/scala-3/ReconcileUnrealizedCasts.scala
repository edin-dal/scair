package scair.transformations.reconcile

import scair.dialects.builtin.*
import scair.ir.*
import scair.transformations.*

object SameType extends RewritePattern {

  override def match_and_rewrite(
      op: Operation,
      rewriter: PatternRewriter
  ): Unit = {
    op match {
      case x: UnrealizedConversionCastOp
        if x.operands.size == 1 && x.results.size == 1 && x.operands.head.typ == x.results.head.typ =>
          rewriter.replace_op(op, Seq(), Some(Seq(x.operands.head)))
      case _ => ()
    }
  }

}

object ReconcileUnrealizedCasts extends ModulePass {
  override val name = "reconcile-unrealized-casts"

  override def transform(op: Operation): Operation = {
    val prw = new PatternRewriteWalker(GreedyRewritePatternApplier(Seq(SameType)))
    prw.rewrite_op(op)

    return op
  }

}

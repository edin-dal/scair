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
      case cast: UnrealizedConversionCastOp
          if cast.operands.size == 1 && cast.results.size == 1 && cast.operands.head.typ == cast.results.head.typ =>
        rewriter.replace_op(op, Seq(), Some(Seq(cast.operands.head)))
      case _ => ()
    }
  }

}

object Unused extends RewritePattern {

  override def match_and_rewrite(
      op: Operation,
      rewriter: PatternRewriter
  ): Unit = {
    op match {
      case cast: UnrealizedConversionCastOp
          if cast.results.forall(_.uses.isEmpty) =>
        rewriter.erase_op(cast)
      case _ => ()
    }
  }

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

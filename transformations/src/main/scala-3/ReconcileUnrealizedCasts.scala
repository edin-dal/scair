package scair.transformations.cdt

import scair.dialects.builtin.*
import scair.ir.*
import scair.transformations.*

object AddDummyAttributeToDict extends RewritePattern {

  override def match_and_rewrite(
      op: Operation,
      rewriter: PatternRewriter
  ): Unit = {
    op match {
      case x:  =>
        x.attributes += ("dummy" -> StringData("UnregDumDum"))
      case d =>
        d.attributes += ("dummy" -> StringData("dumdum"))
    }
    rewriter.has_done_action = true
  }

}

object TestReplacementPass extends ModulePass {
  override val name = "test-rep-pass"

  override def transform(op: Operation): Operation = {
    val prw = new PatternRewriteWalker(TestReplacingDummyOperation)
    prw.rewrite_op(op)

    return op
  }

}

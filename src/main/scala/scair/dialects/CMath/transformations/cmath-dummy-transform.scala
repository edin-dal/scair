package scair.dialects.CMath.transformations.cdt

import scair.{Operation, MLContext}
import scair.dialects.builtin.StringAttribute
import scair.dialects.CMath.cmath.{Mul, Norm}
import scair.transformations.{
  ModulePass,
  PatternRewriter,
  RewritePattern,
  PatternRewriteWalker
}
import scair.UnregisteredOperation

object AddDummyAttributeToDict extends RewritePattern {
  override def match_and_rewrite(
      op: Operation,
      rewriter: PatternRewriter
  ): Unit = {
    op match {
      case x: UnregisteredOperation =>
        x.dictionaryAttributes += ("dummy" -> StringAttribute("UnregDumDum"))
      case y: Mul =>
        y.dictionaryAttributes += ("dummy" -> StringAttribute("MulDumDum"))
      case z: Norm =>
        z.dictionaryAttributes += ("dummy" -> StringAttribute("NormDumDum"))
      case d =>
        d.dictionaryAttributes += ("dummy" -> StringAttribute("dumdum"))
    }
    rewriter.has_done_action = true
  }
}

object DummyPass extends ModulePass {
  override val name = "dummy-pass"

  override def transform(op: Operation): Operation = {
    val prw = new PatternRewriteWalker(AddDummyAttributeToDict)
    prw.rewrite_op(op)

    return op
  }
}

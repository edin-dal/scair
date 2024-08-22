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

object TestInsertingDummyOperation extends RewritePattern {

  def defDum(name: String) = new UnregisteredOperation(name)

  override def match_and_rewrite(
      op: Operation,
      rewriter: PatternRewriter
  ): Unit = {

    (op.name == "tobereplaced") && (op.results.length == 0) match {
      case true =>
        rewriter.insert_op_before_matched_op(
          Seq(
            defDum("dummy1"),
            defDum("dummy2"),
            defDum("dummy3"),
            defDum("dummy4")
          )
        )
        rewriter.erase_matched_op()
      case false =>
        op.dictionaryAttributes += ("replaced" -> StringAttribute("false"))
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

object TestInsertionPass extends ModulePass {
  override val name = "test-ins-pass"

  override def transform(op: Operation): Operation = {
    val prw = new PatternRewriteWalker(TestInsertingDummyOperation)
    prw.rewrite_op(op)

    return op
  }
}

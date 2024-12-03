package scair.transformations.cdt

import scair.ir._
import scair.dialects.builtin.StringData
import scair.dialects.cmath.{Mul, Norm}
import scair.transformations.{
  ModulePass,
  PatternRewriter,
  RewritePattern,
  PatternRewriteWalker
}
import scair.MLContext

object AddDummyAttributeToDict extends RewritePattern {
  override def match_and_rewrite(
      op: Operation,
      rewriter: PatternRewriter
  ): Unit = {
    op match {
      case x: UnregisteredOperation =>
        x.dictionaryAttributes += ("dummy" -> StringData("UnregDumDum"))
      case y: Mul =>
        y.dictionaryAttributes += ("dummy" -> StringData("MulDumDum"))
      case z: Norm =>
        z.dictionaryAttributes += ("dummy" -> StringData("NormDumDum"))
      case d =>
        d.dictionaryAttributes += ("dummy" -> StringData("dumdum"))
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
        op.dictionaryAttributes += ("replaced" -> StringData("false"))
    }

    rewriter.has_done_action = true
  }
}

object TestReplacingDummyOperation extends RewritePattern {

  val val1 = Value[Attribute](StringData("replaced(i32)"))
  val val2 = Value[Attribute](StringData("replaced(i64)"))

  val op1 =
    new UnregisteredOperation("dummy-op1")
  val op2 =
    new UnregisteredOperation("dummy-op2")
  val op3 =
    new UnregisteredOperation("dummy-return", results = ListType(val1, val2))

  val opReplace = new UnregisteredOperation(
    "replacedOp",
    regions = ListType(
      Region(Seq(Block(operations = ListType(op1, op2, op3))))
    )
  )

  override def match_and_rewrite(
      op: Operation,
      rewriter: PatternRewriter
  ): Unit = {

    (op.name == "tobereplaced") match {
      case true =>
        rewriter.replace_op(
          op,
          opReplace,
          Some(op3.results.toSeq)
        )
      case false =>
        op.dictionaryAttributes += ("replaced" -> StringData("false"))
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

object TestReplacementPass extends ModulePass {
  override val name = "test-rep-pass"

  override def transform(op: Operation): Operation = {
    val prw = new PatternRewriteWalker(TestReplacingDummyOperation)
    prw.rewrite_op(op)

    return op
  }
}

package scair.transformations.cdt

import scair.dialects.builtin.StringData
import scair.ir.*
import scair.transformations.*

val AddDummyAttributeToDict = pattern {
  case x: UnregisteredOperation =>
    x.updated(attributes =
      x.attributes + ("dummy" -> StringData("UnregDumDum"))
    )
  case d =>
    d.updated(attributes = d.attributes + ("dummy" -> StringData("dumdum")))
}

val TestInsertingDummyOperation = pattern {

  case op if (op.name == "tobereplaced") && (op.results.length == 0) =>
    Seq(
      UnregisteredOperation("dummy1"),
      UnregisteredOperation("dummy2"),
      UnregisteredOperation("dummy3"),
      UnregisteredOperation("dummy4")
    )
  case op
      if (!op.attributes.contains("replaced")) && op.container_block != None =>
    op.updated(attributes = op.attributes + ("replaced" -> StringData("false")))
}

val TestReplacingDummyOperation = pattern {
  case op if (op.name == "tobereplaced") =>
    val op1 =
      UnregisteredOperation("dummy-op1")

    val op2 =
      UnregisteredOperation("dummy-op2")

    val op3 =
      UnregisteredOperation(
        "dummy-return"
      )

    UnregisteredOperation(
      "replacedOp",
      regions = Seq(Region(Seq(Block(operations = Seq(op1, op2, op3))))),
      results = Seq(StringData("replaced(i32)"), StringData("replaced(i64)"))
        .map(Result(_))
    )
  case op
      if (!op.attributes.contains("replaced")) && op.container_block != None =>
    op.updated(attributes = op.attributes + ("replaced" -> StringData("false")))
}

object DummyPass extends ModulePass {
  override val name = "dummy-pass"

  override def transform(op: Operation): Operation = {
    val prw = PatternRewriteWalker(AddDummyAttributeToDict)
    prw.rewrite_op(op)

    return op
  }

}

object TestInsertionPass extends ModulePass {
  override val name = "test-ins-pass"

  override def transform(op: Operation): Operation = {
    val prw = PatternRewriteWalker(TestInsertingDummyOperation)
    prw.rewrite_op(op)

    return op
  }

}

object TestReplacementPass extends ModulePass {
  override val name = "test-rep-pass"

  override def transform(op: Operation): Operation = {
    val prw = PatternRewriteWalker(TestReplacingDummyOperation)
    prw.rewrite_op(op)

    return op
  }

}

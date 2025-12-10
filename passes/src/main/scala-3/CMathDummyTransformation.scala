package scair.passes.cdt

import scair.MLContext
import scair.dialects.builtin.StringData
import scair.ir.*
import scair.transformations.*
import scair.transformations.patterns.*

val AddDummyAttributeToDict = pattern {
  case x: UnregisteredOperation =>
    x.updated(attributes =
      x.attributes.addOne("dummy" -> StringData("UnregDumDum"))
    )
  case d =>
    d.updated(attributes = d.attributes.addOne("dummy" -> StringData("dumdum")))
}

val TestInsertingDummyOperation = pattern {

  case op if (op.name == "tobereplaced") && (op.results.length == 0) =>
    Seq(
      UnregisteredOperation("dummy1")(),
      UnregisteredOperation("dummy2")(),
      UnregisteredOperation("dummy3")(),
      UnregisteredOperation("dummy4")(),
    )
  case op
      if (!op.attributes.contains("replaced")) && op.containerBlock != None =>
    op.updated(attributes =
      op.attributes.addOne("replaced" -> StringData("false"))
    )
}

val TestReplacingDummyOperation = pattern {
  case op if (op.name == "tobereplaced") =>
    val op1 =
      UnregisteredOperation("dummy-op1")()

    val op2 =
      UnregisteredOperation("dummy-op2")()

    val op3 =
      UnregisteredOperation(
        "dummy-return"
      )()

    UnregisteredOperation("replacedOp")(
      regions = Seq(Region(Block(operations = Seq(op1, op2, op3)))),
      results = Seq(StringData("replaced(i32)"), StringData("replaced(i64)"))
        .map(Result(_)),
    )
  case op
      if (!op.attributes.contains("replaced")) && op.containerBlock != None =>
    op.updated(attributes =
      op.attributes.addOne("replaced" -> StringData("false"))
    )
}

final class DummyPass(ctx: MLContext) extends WalkerPass(ctx):
  override val name = "dummy-pass"

  override final val walker = PatternRewriteWalker(AddDummyAttributeToDict)

final class TestInsertionPass(ctx: MLContext) extends WalkerPass(ctx):
  override val name = "test-ins-pass"

  override final val walker = PatternRewriteWalker(TestInsertingDummyOperation)

final class TestReplacementPass(ctx: MLContext) extends WalkerPass(ctx):
  override val name = "test-rep-pass"

  override final val walker = PatternRewriteWalker(TestReplacingDummyOperation)

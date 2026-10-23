package scair.passes.cdt

import scair.MLContext
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
      UnregisteredOperation("dummy1")(),
      UnregisteredOperation("dummy2")(),
      UnregisteredOperation("dummy3")(),
      UnregisteredOperation("dummy4")(),
    )
  case op
      if (!op.attributes.contains("replaced")) && op.containerBlock != None =>
    op.updated(attributes = op.attributes + ("replaced" -> StringData("false")))
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
    op.updated(attributes = op.attributes + ("replaced" -> StringData("false")))
}

final case class DummyPass() extends WalkerPass derives PassDefs:

  override def walker(using MLContext) = PatternRewriteWalker(
    AddDummyAttributeToDict
  )

@passName("test-ins-pass")
final case class TestInsertionPass() extends WalkerPass derives PassDefs:

  override def walker(using MLContext) = PatternRewriteWalker(
    TestInsertingDummyOperation
  )

@passName("test-rep-pass")
final case class TestReplacementPass() extends WalkerPass derives PassDefs:

  override def walker(using MLContext) = PatternRewriteWalker(
    TestReplacingDummyOperation
  )

enum TestOrder:
  case TopDown, BottomUp

/** Records its own options on the operation it runs on, to test option parsing.
  */
final case class TestPassOptions(
    number: Int = 1,
    ratio: Double = 0.5,
    label: String = "default",
    flag: Boolean = false,
    maybe: Option[Int] = None,
    sizes: Seq[Int] = Seq(),
    order: TestOrder = TestOrder.TopDown,
) extends ModulePass derives PassDefs:

  override def transform(op: Operation)(using MLContext): Operation =
    op.attributes += "test.options" -> StringData(
      PassDefs[TestPassOptions].show(this)
    )
    op

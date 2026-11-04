package scair.passes.cdt

import scair.MLContext
import scair.dialects.builtin.*
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

final case class DummyPass() extends WalkerPass derives PassCompanion:

  override def walker(using MLContext) = PatternRewriteWalker(
    AddDummyAttributeToDict
  )

final case class TestInsertionPass() extends WalkerPass derives PassCompanion:

  override def walker(using MLContext) = PatternRewriteWalker(
    TestInsertingDummyOperation
  )

final case class TestReplacementPass() extends WalkerPass derives PassCompanion:

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
    wide: Long = 0L,
    big: BigInt = BigInt(0),
) extends ModulePass derives PassCompanion:

  override def transform(op: Operation)(using MLContext): Operation =
    op.attributes ++= Map(
      "test.number" -> IntData(number),
      "test.ratio" -> FloatData(ratio),
      "test.label" -> StringData(label),
      "test.flag" -> StringData(flag.toString),
      "test.maybe" -> StringData(maybe.toString),
      "test.sizes" -> ArrayAttribute(sizes.map(v => IntData(v))*),
      "test.order" -> StringData(order.toString),
      "test.wide" -> IntData(wide),
      "test.big" -> IntData(big),
    )
    op

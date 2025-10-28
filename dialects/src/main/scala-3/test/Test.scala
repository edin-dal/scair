package scair.dialects.test

import scair.ir.*

case class TestOp(
    override val operands: Seq[Value[Attribute]] = Seq(),
    override val successors: Seq[Block] = Seq(),
    override val results: Seq[Result[Attribute]] = Seq(),
    override val regions: Seq[Region] = Seq(),
    override val properties: Map[String, Attribute] =
      Map.empty[String, Attribute],
    override val attributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends BaseOperation(
      name = "test.op",
      operands,
      successors,
      results,
      regions,
      properties,
      attributes
    )

object TestOp extends OperationCompanion:
  override def name = "test.op"

val Test: Dialect =
  new Dialect(
    operations = Seq(TestOp),
    attributes = Seq()
  )

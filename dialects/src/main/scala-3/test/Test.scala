package scair.dialects.test

import scair.ir.*

case class TestOp(
    override val operands: ListType[Value[Attribute]] = ListType(),
    override val successors: ListType[Block] = ListType(),
    results_types: ListType[Attribute] = ListType(),
    override val regions: ListType[Region] = ListType(),
    override val properties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val attributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends BaseOperation(
      name = "test.op",
      operands,
      successors,
      results_types,
      regions,
      properties,
      attributes
    )

object TestOp extends OperationCompanion {
  override def name = "test.op"
}

val Test: Dialect =
  new Dialect(
    operations = Seq(TestOp),
    attributes = Seq()
  )

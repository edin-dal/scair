package scair.dialects.test

import scair.ir.*

case class TestOp(
    override val operands: ListType[Value[Attribute]] = ListType(),
    override val successors: ListType[Block] = ListType(),
    results_types: ListType[Attribute] = ListType(),
    override val regions: ListType[Region] = ListType(),
    override val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends RegisteredOperation(
      name = "test.op",
      operands,
      successors,
      results_types,
      regions,
      dictionaryProperties,
      dictionaryAttributes
    )

object TestOp extends OperationObject {
  override def name = "test.op"
  override def factory = TestOp.apply
}

val Test: Dialect =
  new Dialect(
    operations = Seq(TestOp),
    attributes = Seq()
  )

package scair.dialects.test

import scair.clair.macros.summonDialect
import scair.ir.*

object TestOp extends OperationCompanion[TestOp]:
  override def name: String = "test.op"

given OperationCompanion[TestOp] = TestOp

case class TestOp(
    override val operands: Seq[Operand[Attribute]] = Seq(),
    override val successors: Seq[Successor] = Seq(),
    override val results: Seq[Result[Attribute]] = Seq(),
    override val regions: Seq[Region] = Seq(),
    override val properties: Map[String, Attribute] = Map
      .empty[String, Attribute],
    override val attributes: DictType[String, Attribute] = DictType
      .empty[String, Attribute],
) extends Operation:
  override def name = "test.op"

  override def updated(
      operands: Seq[Value[Attribute]] = operands,
      successors: Seq[Block] = successors,
      results: Seq[Result[Attribute]] = results.map(_.typ).map(Result(_)),
      regions: Seq[Region] = detachedRegions,
      properties: Map[String, Attribute] = properties,
      attributes: DictType[String, Attribute] = attributes,
  ) =
    TestOp(
      operands,
      successors,
      results,
      regions,
      properties,
      attributes,
    )

val Test: Dialect = summonDialect[EmptyTuple, Tuple1[TestOp]]

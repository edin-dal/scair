import scair.ir._

object NameOp extends DialectOperation {
  override def name = "dialect.name2"
  override def factory = NameOp.apply
}

case class NameOp(
    override val operands: ListType[Value[Attribute]] = ListType(),
    override val successors: ListType[Block] = ListType(),
    override val results: ListType[Value[Attribute]] = ListType(),
    override val regions: ListType[Region] = ListType(),
    override val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends RegisteredOperation(name = "dialect.name2") {

  override def custom_verify(): Unit =
    if (operands.length != 3) then
      throw new Exception("Expected 3 operands, got operands.length")
    if (results.length != 0) then
      throw new Exception("Expected 0 results, got results.length")
    if (regions.length != 1) then
      throw new Exception("Expected 1 regions, got regions.length")
    if (successors.length != 1) then
      throw new Exception("Expected 1 successors, got successors.length")
    if (dictionaryProperties.size != 0) then
      throw new Exception(
        "Expected 0 properties, got dictionaryProperties.size"
      )
    if (dictionaryAttributes.size != 0) then
      throw new Exception(
        "Expected 0 attributes, got dictionaryAttributes.size"
      )

  def map: Value[Attribute] = operands(0)
  def map_=(value: Value[Attribute]): Unit = { operands(0) = value }

  def map2: Value[Attribute] = operands(1)
  def map2_=(value: Value[Attribute]): Unit = { operands(1) = value }

  def map3: Value[Attribute] = operands(2)
  def map3_=(value: Value[Attribute]): Unit = { operands(2) = value }

  def testregion: Region = regions(0)
  def testregion_=(value: Region): Unit = { regions(0) = value }

  def testsuccessor: Block = successors(0)
  def testsuccessor_=(value: Block): Unit = { successors(0) = value }

}

object NameAttr extends DialectAttribute {
  override def name = "dialect.name1"
  override def factory = NameAttr.apply
}

case class NameAttr(override val parameters: Seq[Attribute])
    extends ParametrizedAttribute(
      name = "dialect.name1",
      parameters = parameters
    )
    with TypeAttribute {
  override def custom_verify(): Unit =
    if (parameters.length != 0) then
      throw new Exception("Expected 0 parameters, got parameters.length")
}

val dialect: Dialect = new Dialect(
  operations = Seq(NameOp),
  attributes = Seq(NameAttr)
)

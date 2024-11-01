package scair.clair

import scala.collection.mutable

// ██╗ ██████╗░
// ██║ ██╔══██╗
// ██║ ██████╔╝
// ██║ ██╔══██╗
// ██║ ██║░░██║
// ╚═╝ ╚═╝░░╚═╝

// ███████╗ ██╗░░░░░ ███████╗ ███╗░░░███╗ ███████╗ ███╗░░██╗ ████████╗ ░██████╗
// ██╔════╝ ██║░░░░░ ██╔════╝ ████╗░████║ ██╔════╝ ████╗░██║ ╚══██╔══╝ ██╔════╝
// █████╗░░ ██║░░░░░ █████╗░░ ██╔████╔██║ █████╗░░ ██╔██╗██║ ░░░██║░░░ ╚█████╗░
// ██╔══╝░░ ██║░░░░░ ██╔══╝░░ ██║╚██╔╝██║ ██╔══╝░░ ██║╚████║ ░░░██║░░░ ░╚═══██╗
// ███████╗ ███████╗ ███████╗ ██║░╚═╝░██║ ███████╗ ██║░╚███║ ░░░██║░░░ ██████╔╝
// ╚══════╝ ╚══════╝ ╚══════╝ ╚═╝░░░░░╚═╝ ╚══════╝ ╚═╝░░╚══╝ ░░░╚═╝░░░ ╚═════╝░

/*≡≡=---=≡≡≡=---=≡≡*\
||      UTILS      ||
\*≡==----=≡=----==≡*/

val DictType = mutable.Map
type DictType[A, B] = mutable.Map[A, B]

val ListType = mutable.ListBuffer
type ListType[A] = mutable.ListBuffer[A]

/*≡≡=--=≡≡≡=--=≡≡*\
||     TYPES     ||
\*≡==---=≡=---==≡*/

abstract class Type(val id: String) {
  def get_import(): String
}

case class RegularType(val dialect: String, override val id: String)
    extends Type(id) {
  override def get_import(): String = s"import scair.dialects.${dialect}._\n"
}

/*≡≡=---==≡≡≡==---=≡≡*\
||    CONSTRAINTS    ||
\*≡==----==≡==----==≡*/

abstract class ConstraintDef {
  def print(indent: Int): String
  def get_imports(): String
}

case class Equal(val typ: Type) extends ConstraintDef {
  override def print(indent: Int): String =
    s"val ${typ.id.toLowerCase()}_check = EqualAttr(${typ.id})\n"

  override def get_imports(): String = typ.get_import()
}
case class Base(val typ: Type) extends ConstraintDef {
  override def print(indent: Int): String =
    s"val ${typ.id.toLowerCase()}_check = BaseAttr[${typ.id}]()\n"

  override def get_imports(): String = typ.get_import()
}
case class AnyOf(val typ: Seq[Type]) extends ConstraintDef {
  override def print(indent: Int): String =
    s"val ${(for (x <- typ) yield x.id).mkString("_").toLowerCase()}_check = AnyOf(Seq(${(for (x <- typ)
        yield x.id).mkString(", ")}))\n"

  override def get_imports(): String =
    (for (x <- typ) yield x.get_import()).mkString("\n")
}

/*≡≡=---===≡≡≡≡===---=≡≡*\
||  TYPES & CONTAINERS  ||
\*≡==----===≡≡===----==≡*/

abstract class OpInput {}

case class OperandDef(val id: String, val const: ConstraintDef)
    extends OpInput {}
case class ResultDef(val id: String, val const: ConstraintDef)
    extends OpInput {}
case class RegionDef(val no: Int) extends OpInput {}
case class SuccessorDef(val no: Int) extends OpInput {}
case class OpPropertyDef(val id: String, val const: ConstraintDef)
    extends OpInput {}
case class OpAttributeDef(val id: String, val const: ConstraintDef)
    extends OpInput {}

case class DialectDef(
    val name: String,
    val operations: ListType[OperationDef] = ListType(),
    val attributes: ListType[AttributeDef] = ListType()
) {
  def print(indent: Int): String = s"""
import scair.{
  ListType,
  DictType,
  RegisteredOperation,
  Region,
  Block,
  Value,
  Attribute,
  TypeAttribute,
  ParametrizedAttribute,
  DialectAttribute,
  DialectOperation,
  Dialect
}
  """ +
    (operations.map(_.print(0)) ++ attributes.map(_.print(0)))
      .mkString("\n") + s"""
val $name: Dialect = new Dialect(
  operations = Seq(${operations.map(_.className).mkString(", ")}),
  attributes = Seq(${attributes.map(_.className).mkString(", ")})
)
  """
}

/*≡≡=---=≡≡≡=---=≡≡*\
||   IR ELEMENTS   ||
\*≡==----=≡=----==≡*/

case class OperationDef(
    val name: String,
    val className: String,
    val operands: Seq[OperandDef],
    val results: Seq[ResultDef],
    val regions_no: RegionDef,
    val successors_no: SuccessorDef,
    val OpProperty: Seq[OpPropertyDef],
    val OpAttribute: Seq[OpAttributeDef]
) {

  def get_imports(): Set[String] = {
    ((for (x <- operands) yield x.const.get_imports()) ++
      (for (x <- results) yield x.const.get_imports()) ++
      (for (x <- OpProperty) yield x.const.get_imports()) ++
      (for (x <- OpAttribute) yield x.const.get_imports())).toSet
  }

  def constraint_printer(indent: Int): String = {
    ((for (x <- operands) yield x.const.print(indent)) ++
      (for (x <- results) yield x.const.print(indent)) ++
      (for (x <- OpProperty) yield x.const.print(indent)) ++
      (for (x <- OpAttribute) yield x.const.print(indent))).toSet.mkString("\n")
  }

  def print(indent: Int): String = s"""
object $className extends DialectOperation {
  override def name = "$name"
  override def factory = $className.apply
}

case class $className(
    override val operands: ListType[Value[Attribute]] = ListType(),
    override val successors: ListType[Block] = ListType(),
    override val results: ListType[Value[Attribute]] = ListType(),
    override val regions: ListType[Region] = ListType(),
    override val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends RegisteredOperation(name = "$name") {

  override def custom_verify(): Unit = 
    if (operands.length != ${operands.length}) then throw new Exception("Expected ${operands.length} operands, got operands.length") 
    if (results.length != ${results.length}) then throw new Exception("Expected ${results.length} results, got results.length")
    if (regions.length != ${regions_no.no}) then throw new Exception("Expected ${regions_no.no} regions, got regions.length")
    if (successors.length != ${successors_no.no}) then throw new Exception("Expected ${successors_no.no} successors, got successors.length")
    if (dictionaryProperties.size != ${OpProperty.length}) then throw new Exception("Expected ${OpProperty.length} properties, got dictionaryProperties.size")
    if (dictionaryAttributes.size != ${OpAttribute.length}) then throw new Exception("Expected ${OpAttribute.length} attributes, got dictionaryAttributes.size")

}
  """
}

case class AttributeDef(
    val name: String,
    val className: String,
    val parameters: Seq[OperandDef],
    val typee: Int
) {
  def print(indent: Int): String = s"""
object $className extends DialectAttribute {
  override def name = "$name"
  override def factory = $className.apply
}

case class $className(override val parameters: Seq[Attribute]) extends ParametrizedAttribute(name = "$name", parameters = parameters) ${
      if typee != 0 then "with TypeAttribute" else ""
    } {
  override def custom_verify(): Unit = 
    if (parameters.length != ${parameters.length}) then throw new Exception("Expected ${parameters.length} parameters, got parameters.length")
}
  """
}

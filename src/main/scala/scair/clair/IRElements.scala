package scair.clair.ir

import scala.collection.mutable
import scala.compiletime.ops.int

import scair.scairdl.constraints._

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

// abstract class ConstraintDef {
//   def print(indent: Int): String
//   def get_imports(): String
// }

// case class Equal(val typ: Type) extends ConstraintDef {
//   override def print(indent: Int): String =
//     s"val ${typ.id.toLowerCase()}_check = EqualAttr(${typ.id})\n"

//   override def get_imports(): String = typ.get_import()
// }
// case class Base(val typ: Type) extends ConstraintDef {
//   override def print(indent: Int): String =
//     s"val ${typ.id.toLowerCase()}_check = BaseAttr[${typ.id}]()\n"

//   override def get_imports(): String = typ.get_import()
// }
// case class AnyOf(val typ: Seq[Type]) extends ConstraintDef {
//   override def print(indent: Int): String =
//     s"val ${(for (x <- typ) yield x.id).mkString("_").toLowerCase()}_check = AnyOf(Seq(${(for (x <- typ)
//         yield x.id).mkString(", ")}))\n"

//   override def get_imports(): String =
//     (for (x <- typ) yield x.get_import()).mkString("\n")
// }

/*≡≡=---===≡≡≡≡===---=≡≡*\
||  TYPES & CONTAINERS  ||
\*≡==----===≡≡===----==≡*/

abstract class OpInput {}

case class OperandDef(val id: String, val const: IRDLConstraint)
    extends OpInput {}
case class ResultDef(val id: String, val const: IRDLConstraint)
    extends OpInput {}
case class RegionDef(val id: String) extends OpInput {}
case class SuccessorDef(val id: String) extends OpInput {}
case class OpPropertyDef(val id: String, val const: IRDLConstraint)
    extends OpInput {}
case class OpAttributeDef(val id: String, val const: IRDLConstraint)
    extends OpInput {}

case class DialectDef(
    val name: String,
    val operations: ListType[OperationDef] = ListType(),
    val attributes: ListType[AttributeDef] = ListType()
) {
  def print(indent: Int): String = s"""
import scair.ir._
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
    val operands: Seq[OperandDef] = Seq(),
    val results: Seq[ResultDef] = Seq(),
    val regions: Seq[RegionDef] = Seq(),
    val successors: Seq[SuccessorDef] = Seq(),
    val OpProperty: Seq[OpPropertyDef] = Seq(),
    val OpAttribute: Seq[OpAttributeDef] = Seq()
) {

  // def get_imports(): Set[String] = {
  //   ((for (x <- operands) yield x.const.get_imports()) ++
  //     (for (x <- results) yield x.const.get_imports()) ++
  //     (for (x <- OpProperty) yield x.const.get_imports()) ++
  //     (for (x <- OpAttribute) yield x.const.get_imports())).toSet
  // }

  // def constraint_printer(implicit indent: Int): String = {
  //   ((for (x <- operands) yield x.const.print(indent)) ++
  //     (for (x <- results) yield x.const.print(indent)) ++
  //     (for (x <- OpProperty) yield x.const.print(indent)) ++
  //     (for (x <- OpAttribute) yield x.const.print(indent))).toSet.mkString("\n")
  // }

  def print_getters(implicit indent: Int): String = {
    ((for (odef, i) <- operands.zipWithIndex
    yield s"  def ${odef.id}: Value[Attribute] = operands($i)\n" +
      s"  def ${odef.id}_=(value: Value[Attribute]): Unit = {operands($i) = value}\n") ++
      (for (rdef, i) <- results.zipWithIndex
      yield s"  def ${rdef.id}: Value[Attribute] = results($i)\n" +
        s"  def ${rdef.id}_=(value: Value[Attribute]): Unit = {results($i) = value}\n") ++
      (for (rdef, i) <- regions.zipWithIndex
      yield s"  def ${rdef.id}: Region = regions($i)\n" +
        s"  def ${rdef.id}_=(value: Region): Unit = {regions($i) = value}\n") ++
      (for (sdef, i) <- successors.zipWithIndex
      yield s"  def ${sdef.id}: Block = successors($i)\n" +
        s"  def ${sdef.id}_=(value: Block): Unit = {successors($i) = value}\n") ++
      (for pdef <- OpProperty
      yield s"  def ${pdef.id}: Attributes = dictionaryProperties(${pdef.id})\n" +
        s"  def ${pdef.id}_=(value: Attributes): Unit = {dictionaryProperties(${pdef.id}) = value}\n") ++
      (for adef <- OpAttribute
      yield s"  def ${adef.id}: Attributes = dictionaryAttributes(${adef.id})\n" +
        s"  def ${adef.id}_=(value: Attributes): Unit = {dictionaryAttributes(${adef.id}) = value}\n"))
      .mkString("\n")

  }

  def print(implicit indent: Int): String = s"""
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
    if (regions.length != ${regions.length}) then throw new Exception("Expected ${regions.length} regions, got regions.length")
    if (successors.length != ${successors.length}) then throw new Exception("Expected ${successors.length} successors, got successors.length")
    if (dictionaryProperties.size != ${OpProperty.length}) then throw new Exception("Expected ${OpProperty.length} properties, got dictionaryProperties.size")
    if (dictionaryAttributes.size != ${OpAttribute.length}) then throw new Exception("Expected ${OpAttribute.length} attributes, got dictionaryAttributes.size")

${print_getters(indent + 1)}
}
  """
}

case class AttributeDef(
    val name: String,
    val className: String,
    val parameters: Seq[OperandDef] = Seq(),
    val typee: Int = 0
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

package scair.clair.ir

import scala.collection.mutable
import scala.compiletime.ops.int

import scair.scairdl.constraints._
import scair.dialects.builtin._
import scair.transformations.InsertPoint.after

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

// RETIRED TO A HOLIDAY RESORT IN NORTHERN SCOTLAND, POSSIBLY PERMANENTLY :')

/*≡≡=---===≡≡≡≡===---=≡≡*\
||  TYPES & CONTAINERS  ||
\*≡==----===≡≡===----==≡*/

abstract class OpInput {}

// TODO: Add support for optionals AFTER variadic support is laid out
// It really just adds cognitive noise otherwise IMO. The broader structure and logic is exactly the same.
// (An Optional structurally is just a Variadic capped at one.)
enum Variadicity {
  case Single, Variadic
}

case class OperandDef(
    val id: String,
    val const: IRDLConstraint = AnyAttr,
    val variadicity: Variadicity = Variadicity.Single
) extends OpInput {}
case class ResultDef(
    val id: String,
    val const: IRDLConstraint = AnyAttr
) extends OpInput {}
case class RegionDef(
    val id: String
) extends OpInput {}
case class SuccessorDef(
    val id: String
) extends OpInput {}
case class OpPropertyDef(val id: String, val const: IRDLConstraint = AnyAttr)
    extends OpInput {}
case class OpAttributeDef(val id: String, val const: IRDLConstraint = AnyAttr)
    extends OpInput {}

case class DialectDef(
    val name: String,
    val operations: ListType[OperationDef] = ListType(),
    val attributes: ListType[AttributeDef] = ListType()
) {
  def print(indent: Int): String = s"""
import scair.ir._
import scair.dialects.builtin._
import scair.scairdl.constraints._
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

  def operand_segment_sizes_helper: String =
    s"""def operandSegmentSizes: Seq[Int] =
    if (!dictionaryProperties.contains("operandSegmentSizes")) then throw new Exception("Expected operandSegmentSizes property")
    val operandSegmentSizes_attr = dictionaryProperties("operandSegmentSizes") match {
      case right: DenseArrayAttr => right
      case _ => throw new Exception("Expected operandSegmentSizes to be a DenseArrayAttr")
    }
    ${ParametrizedAttrConstraint[DenseArrayAttr](
        Seq(
          EqualAttr(IntegerType(IntData(32), Signless)),
          AllOf(
            Seq(
              BaseAttr[IntegerAttr](),
              ParametrizedAttrConstraint[IntegerAttr](
                Seq(
                  BaseAttr[IntData](),
                  EqualAttr(IntegerType(IntData(32), Signless))
                )
              )
            )
          )
        )
      )}.verify(operandSegmentSizes_attr, ConstraintContext())
    if (operandSegmentSizes_attr.length != ${operands.length}) then throw new Exception(s"Expected operandSegmentSizes to have ${operands.length} elements, got $${operandSegmentSizes_attr.length}")
    
    for (s <- operandSegmentSizes_attr) yield s match {
      case right: IntegerAttr => right.value.data.toInt
      case _ => throw new Exception("Unreachable exception as per above constraint check.")
    }
    """

  def helpers(implicit indent: Int): String = s"""
  ${if (n_variadic_operands > 1) then operand_segment_sizes_helper else ""}
  """

  def print_constr_defs(implicit indent: Int): String = {
    val deff = { (x: String, y: IRDLConstraint) =>
      s"  val ${x}_constr = ${y}"
    }
    ((for (odef <- operands) yield deff(odef.id, odef.const)) ++
      (for (rdef <- results) yield deff(rdef.id, rdef.const)) ++
      (for (pdef <- OpProperty) yield deff(pdef.id, pdef.const)) ++
      (for (adef <- OpAttribute)
        yield deff(adef.id, adef.const))).mkString("\n")
  }

  def n_variadic_operands: Int =
    operands.filter(_.variadicity != Variadicity.Single).length

  def single_operand_accessor(name: String, index: String) =
    s"  def ${name}: Value[Attribute] = operands($index)\n" +
      s"  def ${name}_=(value: Value[Attribute]): Unit = {operands($index) = value}\n"

  def segmented_single_operand_accessor(name: String, index: String) =
    single_operand_accessor(
      name,
      s"operandSegmentSizes.slice(0, $index).reduce(_ + _)"
    )

  def variadic_operand_accessor(name: String, from: String, to: String) =
    s"""  def ${name}: Seq[Value[Attribute]] = {
      val from = $from
      val to = $to
      operands.slice(from, to).toSeq
  }
  def ${name}_=(values: Seq[Value[Attribute]]): Unit = {
    val from = $from
    val to = $to
    val diff = values.length - (to - from)
    for (value, i) <- (values ++ operands.slice(to, operands.length)).zipWithIndex do
      operands(from + i) = value
    if (diff < 0)
      operands.trimEnd(-diff)
  }\n\n"""

  def segmented_variadic_operand_accessor(name: String, index: String) =
    variadic_operand_accessor(
      name,
      s"operandSegmentSizes.slice(0, $index).reduce(_ + _)",
      s"from + operandSegmentSizes($index)"
    )

  def operands_accessors(implicit indent: Int): Seq[String] = {
    n_variadic_operands match {
      case 0 =>
        for ((odef, i) <- operands.zipWithIndex)
          yield single_operand_accessor(odef.id, i.toString)
      case 1 => {
        val variadic_index =
          operands.indexWhere(_.variadicity != Variadicity.Single);
        ((
          for (odef, i) <- operands.slice(0, variadic_index).zipWithIndex
          yield single_operand_accessor(odef.id, i.toString)
        )
          :+
            (variadic_operand_accessor(
              operands(variadic_index).id,
              variadic_index.toString,
              s"operands.length - ${operands.length - variadic_index - 1}"
            ))) ++
          (for (
            (odef, i) <- operands.zipWithIndex
              .slice(
                variadic_index + 1,
                operands.length
              )
          )
            yield single_operand_accessor(
              odef.id,
              s"operands.length - ${operands.length - i}"
            ))

      }
      case _: Int => {
        for ((odef, i) <- operands.zipWithIndex)
          yield odef.variadicity match {
            case Variadicity.Single =>
              segmented_single_operand_accessor(odef.id, i.toString)
            case Variadicity.Variadic =>
              segmented_variadic_operand_accessor(odef.id, i.toString)
          }
      }
    }

  }

  def accessors(implicit indent: Int): String = {
    (operands_accessors ++
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

  def constraints_verification(implicit indent: Int): String = {
    val ver = { (x: String) =>
      s"${x}_constr.verify($x.typ, verification_context)"
    }
    ((for (odef <- operands) yield ver(odef.id)) ++
      (for (rdef <- results) yield ver(rdef.id)) ++
      (for (pdef <- OpProperty) yield ver(pdef.id)) ++
      (for (adef <- OpAttribute) yield ver(adef.id))).mkString("\n    ")
  }

  def operands_verification(implicit indent: Int): String =
    n_variadic_operands match {
      case 0 =>
        s"""if (operands.length != ${operands.length}) then throw new Exception(s"Expected ${operands.length} operands, got $${operands.length}")"""
      case 1 => {
        s"""if (operands.length < ${operands.length - 1}) then throw new Exception(s"Expected at least ${operands.length - 1} operands, got $${operands.length}")"""
      }
      case _: Int => {
        s"""val operandSegmentSizesSum = operandSegmentSizes.reduce(_ + _)
    if (operandSegmentSizesSum != operands.length) then throw new Exception(s"Expected $${operandSegmentSizesSum} operands, got $${operands.length}")\n""" +
          (for (
            (odef, i) <- operands.zipWithIndex.filter(
              _._1.variadicity == Variadicity.Single
            )
          )
            yield s"""    if operandSegmentSizes($i) != 1 then throw new Exception("operand segment size expected to be 1 for singular operand ${odef.id} at index $i, got $${operandSegmentSizes($i)}")""")
            .mkString("\n")

      }
    }

  def constructs_verification(implicit indent: Int): String = s"""
    ${operands_verification(indent + 1)} 
    if (results.length != ${results.length}) then throw new Exception("Expected ${results.length} results, got results.length")
    if (regions.length != ${regions.length}) then throw new Exception("Expected ${regions.length} regions, got regions.length")
    if (successors.length != ${successors.length}) then throw new Exception("Expected ${successors.length} successors, got successors.length")
    if (dictionaryProperties.size != ${OpProperty.length}) then throw new Exception("Expected ${OpProperty.length} properties, got dictionaryProperties.size")
    if (dictionaryAttributes.size != ${OpAttribute.length}) then throw new Exception("Expected ${OpAttribute.length} attributes, got dictionaryAttributes.size")
"""

  def irdl_verification(implicit indent: Int): String = s"""
  override def custom_verify(): Unit = 
    val verification_context = new ConstraintContext()
    ${constructs_verification(indent + 1)}
    ${constraints_verification(indent + 1)}
"""

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

${helpers(indent + 1)}
${accessors(indent + 1)}
${print_constr_defs(indent + 1)}
${irdl_verification(indent + 1)}

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

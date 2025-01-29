package scair.scairdl.irdef

import scair.dialects.builtin.*
import scair.ir.Attribute
import scair.ir.Operation
import scair.scairdl.constraints.*

import java.io.File
import java.io.PrintStream
import scala.collection.mutable
import scala.reflect.*

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

abstract class EscapeHatch[T: ClassTag] {

  val importt: String =
    s"import ${implicitly[ClassTag[T]].runtimeClass.getName.replace("$", ".")}"

  val name: String = importt.split("\\.").last.split(" ").last
}

class AttrEscapeHatch[T <: Attribute: ClassTag]() extends EscapeHatch[T]

class OpEscapeHatch[T <: Operation: ClassTag]() extends EscapeHatch[T]

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

/*≡≡=---===≡≡≡≡===---=≡≡*\
||  TYPES & CONTAINERS  ||
\*≡==----===≡≡===----==≡*/

abstract class OpInput(val id: String) {

  def def_type: String
  def def_name: String
  def def_field: String

  def single_accessors(index: String) =
    s"  def ${id}: $def_type = $def_field($index)\n" +
      s"  def ${id}_=(new_${def_name}: $def_type): Unit = {$def_field($index) = new_${def_name}}\n"

  def segmented_single_accessors(index: String) =
    single_accessors(
      s"${def_name}SegmentSizes.slice(0, $index).fold(0)(_ + _)"
    )

  def variadic_accessors(from: String, to: String) =
    s"""  def ${id}: $def_type = {
      val from = $from
      val to = $to
      $def_field.slice(from, to).toSeq
  }
  def ${id}_=(new_${def_name}s: $def_type): Unit = {
    val from = $from
    val to = $to
    val diff = new_${def_name}s.length - (to - from)
    for (${def_name}, i) <- (new_${def_name}s ++ $def_field.slice(to, $def_field.length)).zipWithIndex do
      $def_field(from + i) = ${def_name}
    if (diff < 0)
      $def_field.trimEnd(-diff)
  }\n\n"""

  def segmented_variadic_accessors(index: String) =
    variadic_accessors(
      s"${def_name}SegmentSizes.slice(0, $index).fold(0)(_ + _)",
      s"from + ${def_name}SegmentSizes($index)"
    )
}

// TODO: Add support for optionals AFTER variadic support is laid out
// It really just adds cognitive noise otherwise IMO. The broader structure and logic is exactly the same.
// (An Optional structurally is just a Variadic capped at one.)
enum Variadicity {
  case Single, Variadic
}

case class OperandDef(
    override val id: String,
    val const: IRDLConstraint = AnyAttr,
    val variadicity: Variadicity = Variadicity.Single
) extends OpInput {}

case class ResultDef(
    override val id: String,
    val const: IRDLConstraint = AnyAttr,
    val variadicity: Variadicity = Variadicity.Single
) extends OpInput {}

case class RegionDef(
    override val id: String,
    val variadicity: Variadicity = Variadicity.Single
) extends OpInput {}

case class SuccessorDef(
    override val id: String,
    val variadicity: Variadicity = Variadicity.Single
) extends OpInput {}

case class OpPropertyDef(
    override val id: String,
    val const: IRDLConstraint = AnyAttr
) extends OpInput {}

case class OpAttributeDef(
    override val id: String,
    val const: IRDLConstraint = AnyAttr
) extends OpInput(id) {
  def def_field: String = "attributes"
  def def_name: String = "attribute"
  def def_type: String = "Attribute"
}

case class AssemblyFormatDef(
    val format: String
) extends OpInput {}

/*≡≡=---=≡≡≡≡≡=---=≡≡*\
||    DIALECT DEF    ||
\*≡==----=≡≡≡=----==≡*/

object DialectDef {
  def empty: DialectDef = DialectDef("empty")
}

case class DialectDef(
    val name: String,
    val operations: Seq[OperationDef] = Seq(),
    val attributes: Seq[AttributeDef] = Seq(),
    val opHatches: Seq[OpEscapeHatch[_]] = Seq(),
    val attrHatches: Seq[AttrEscapeHatch[_]] = Seq()
) {

  def print(indent: Int): String =
    s"""package scair.dialects.${name.toLowerCase}

import scair.ir._
import scair.dialects.builtin._
import scair.scairdl.constraints._
import scair.scairdl.constraints.attr2constraint"""
      + { for (hatch <- opHatches) yield hatch.importt }
        .mkString("\n", "\n", "\n") +
      { for (hatch <- attrHatches) yield hatch.importt }
        .mkString("", "\n", "\n") +
      (operations.map(_.print(0)) ++ attributes.map(_.print(0)))
        .mkString("\n") + s"""
val ${name}Dialect: Dialect = new Dialect(
  operations = Seq(${(operations.map(_.className) ++ opHatches.map(_.name))
          .mkString(", ")}),
  attributes = Seq(${(attributes.map(_.className) ++ attrHatches.map(_.name))
          .mkString(", ")})
)
  """

}

/*≡≡=---=≡≡≡≡≡=---=≡≡*\
||   OPERATION DEF   ||
\*≡==----=≡≡≡=----==≡*/

case class OperationDef(
    val name: String,
    val className: String,
    val operands: Seq[OperandDef] = Seq(),
    val results: Seq[ResultDef] = Seq(),
    val regions: Seq[RegionDef] = Seq(),
    val successors: Seq[SuccessorDef] = Seq(),
    val OpProperty: Seq[OpPropertyDef] = Seq(),
    val OpAttribute: Seq[OpAttributeDef] = Seq(),
    val assembly_format: Option[String] = None
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

  def result_segment_sizes_helper: String =
    s"""def resultSegmentSizes: Seq[Int] =
    if (!dictionaryProperties.contains("resultSegmentSizes")) then throw new Exception("Expected resultSegmentSizes property")
    val resultSegmentSizes_attr = dictionaryProperties("resultSegmentSizes") match {
      case right: DenseArrayAttr => right
      case _ => throw new Exception("Expected resultSegmentSizes to be a DenseArrayAttr")
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
      )}.verify(resultSegmentSizes_attr, ConstraintContext())
    if (resultSegmentSizes_attr.length != ${results.length}) then throw new Exception(s"Expected resultSegmentSizes to have ${results.length} elements, got $${resultSegmentSizes_attr.length}")

    for (s <- resultSegmentSizes_attr) yield s match {
      case right: IntegerAttr => right.value.data.toInt
      case _ => throw new Exception("Unreachable exception as per above constraint check.")
    }
    """

  def helpers(implicit indent: Int): String = s"""
  ${if (n_variadic_operands > 1) then operand_segment_sizes_helper else ""}
  ${if (n_variadic_results > 1) then result_segment_sizes_helper else ""}
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

  def n_variadic_results: Int =
    results.filter(_.variadicity != Variadicity.Single).length

  def n_variadic_regions: Int =
    regions.filter(_.variadicity != Variadicity.Single).length

  def n_variadic_successors: Int =
    successors.filter(_.variadicity != Variadicity.Single).length

  def operands_accessors(implicit indent: Int): Seq[String] = {
    n_variadic_operands match {
      case 0 =>
        for ((odef, i) <- operands.zipWithIndex)
          yield odef.single_accessors(i.toString)
      case 1 => {
        val variadic_index =
          operands.indexWhere(_.variadicity != Variadicity.Single);
        val variadic = operands(variadic_index)
        ((
          for (odef, i) <- operands.slice(0, variadic_index).zipWithIndex
          yield odef.single_accessors(i.toString)
        )
          :+
            variadic.variadic_accessors(
              variadic_index.toString,
              s"operands.length - ${operands.length - variadic_index - 1}"
            )) ++
          (for (
            (odef, i) <- operands.zipWithIndex
              .slice(
                variadic_index + 1,
                operands.length
              )
          )
            yield odef.single_accessors(
              s"operands.length - ${operands.length - i}"
            ))

      }
      case _: Int => {
        for ((odef, i) <- operands.zipWithIndex)
          yield odef.variadicity match {
            case Variadicity.Single =>
              odef.segmented_single_accessors(i.toString)
            case Variadicity.Variadic =>
              odef.segmented_variadic_accessors(i.toString)
          }
      }
    }
  }

  def results_accessors(implicit indent: Int): Seq[String] = {
    n_variadic_results match {
      case 0 =>
        for ((odef, i) <- results.zipWithIndex)
          yield odef.single_accessors(i.toString)
      case 1 => {
        val variadic_index =
          results.indexWhere(_.variadicity != Variadicity.Single)
        val variadic = results(variadic_index)
        ((
          for (odef, i) <- results.slice(0, variadic_index).zipWithIndex
          yield odef.single_accessors(i.toString)
        )
          :+
            (variadic.variadic_accessors(
              variadic_index.toString,
              s"results.length - ${results.length - variadic_index - 1}"
            ))) ++
          (for (
            (odef, i) <- results.zipWithIndex
              .slice(
                variadic_index + 1,
                results.length
              )
          )
            yield odef.single_accessors(
              s"results.length - ${results.length - i}"
            ))

      }
      case _: Int => {
        for ((odef, i) <- results.zipWithIndex)
          yield odef.variadicity match {
            case Variadicity.Single =>
              odef.segmented_single_accessors(i.toString)
            case Variadicity.Variadic =>
              odef.segmented_variadic_accessors(i.toString)
          }
      }
    }
  }

  def regions_accessors(implicit indent: Int): Seq[String] = {
    n_variadic_regions match {
      case 0 =>
        for ((odef, i) <- regions.zipWithIndex)
          yield odef.single_accessors(i.toString)
      case 1 => {
        val variadic_index =
          regions.indexWhere(_.variadicity != Variadicity.Single)
        val variadic = regions(variadic_index)
        ((
          for (odef, i) <- regions.slice(0, variadic_index).zipWithIndex
          yield odef.single_accessors(i.toString)
        )
          :+
            (variadic.variadic_accessors(
              variadic_index.toString,
              s"regions.length - ${regions.length - variadic_index - 1}"
            ))) ++
          (for (
            (odef, i) <- regions.zipWithIndex
              .slice(
                variadic_index + 1,
                regions.length
              )
          )
            yield odef.single_accessors(
              s"regions.length - ${regions.length - i}"
            ))

      }
      // TODO: Discuss this
      // Multivariadic regions and successors are a thing in xDSL, not in MLIR.
      // Thing is, given the architecture of frameworks following IRDL principles,
      // it might end up *more* trivial to just support them, following the same
      // logic than operands and results.
      // But, we don't need them for MLIR-compatibility. So skipping for now!
      case _: Int =>
        throw new NotImplementedError("Multivariadic regions not implemented")
    }
  }

  def successors_accessors(implicit indent: Int): Seq[String] = {
    n_variadic_successors match {
      case 0 =>
        for ((odef, i) <- successors.zipWithIndex)
          yield odef.single_accessors(i.toString)
      case 1 => {
        val variadic_index =
          successors.indexWhere(_.variadicity != Variadicity.Single)
        val variadic = successors(variadic_index)
        ((
          for (odef, i) <- successors.slice(0, variadic_index).zipWithIndex
          yield odef.single_accessors(i.toString)
        )
          :+
            (variadic.variadic_accessors(
              variadic_index.toString,
              s"successors.length - ${successors.length - variadic_index - 1}"
            ))) ++
          (for (
            (odef, i) <- successors.zipWithIndex
              .slice(
                variadic_index + 1,
                successors.length
              )
          )
            yield odef.single_accessors(
              s"successors.length - ${successors.length - i}"
            ))

      }
      case _: Int =>
        throw new NotImplementedError(
          "Multivariadic successors not implemented"
        )
    }
  }

  def accessors(implicit indent: Int): String = {
    (operands_accessors ++
      results_accessors ++
      regions_accessors ++
      successors_accessors ++
      (for pdef <- OpProperty
      yield s"  def ${pdef.id}: Attribute = dictionaryProperties(\"${pdef.id}\")\n" +
        s"  def ${pdef.id}_=(new_attribute: Attribute): Unit = {dictionaryProperties(\"${pdef.id}\") = new_attribute}\n") ++
      (for adef <- OpAttribute
      yield s"  def ${adef.id}: Attribute = dictionaryAttributes(\"${adef.id}\")\n" +
        s"  def ${adef.id}_=(new_attribute: Attribute): Unit = {dictionaryAttributes(\"${adef.id}\") = new_attribute}\n"))
      .mkString("\n")

  }

  def constraints_verification(implicit indent: Int): String = {
    val verify_value = { (x: String) =>
      s"${x}_constr.verify($x.typ, verification_context)"
    }
    val verify_attribute = { (x: String) =>
      s"${x}_constr.verify($x, verification_context)"
    }
    ((for (odef <- operands) yield verify_value(odef.id)) ++
      (for (rdef <- results) yield verify_value(rdef.id)) ++
      (for (pdef <- OpProperty) yield verify_attribute(pdef.id)) ++
      (for (adef <- OpAttribute)
        yield verify_attribute(adef.id))).mkString("\n    ")
  }

  def operands_verification(implicit indent: Int): String =
    n_variadic_operands match {
      case 0 =>
        s"""if (operands.length != ${operands.length}) then throw new Exception(s"Expected ${operands.length} operands, got $${operands.length}")"""
      case 1 => {
        s"""if (operands.length < ${operands.length - 1}) then throw new Exception(s"Expected at least ${operands.length - 1} operands, got $${operands.length}")"""
      }
      case _: Int => {
        s"""val operandSegmentSizesSum = operandSegmentSizes.fold(0)(_ + _)
    if (operandSegmentSizesSum != operands.length) then throw new Exception(s"Expected $${operandSegmentSizesSum} operands, got $${operands.length}")\n""" +
          (for (
            (odef, i) <- operands.zipWithIndex.filter(
              _._1.variadicity == Variadicity.Single
            )
          )
            yield s"""    if operandSegmentSizes($i) != 1 then throw new Exception(s"operand segment size expected to be 1 for singular operand ${odef.id} at index $i, got $${operandSegmentSizes($i)}")""")
            .mkString("\n")

      }
    }

  def results_verification(implicit indent: Int): String =
    n_variadic_results match {
      case 0 =>
        s"""if (results.length != ${results.length}) then throw new Exception(s"Expected ${results.length} results, got $${results.length}")"""
      case 1 => {
        s"""if (results.length < ${results.length - 1}) then throw new Exception(s"Expected at least ${results.length - 1} results, got $${results.length}")"""
      }
      case _: Int => {
        s"""val resultSegmentSizesSum = resultSegmentSizes.fold(0)(_ + _)
    if (resultSegmentSizesSum != results.length) then throw new Exception(s"Expected $${resultSegmentSizesSum} results, got $${results.length}")\n""" +
          (for (
            (odef, i) <- results.zipWithIndex.filter(
              _._1.variadicity == Variadicity.Single
            )
          )
            yield s"""    if resultSegmentSizes($i) != 1 then throw new Exception(s"result segment size expected to be 1 for singular result ${odef.id} at index $i, got $${resultSegmentSizes($i)}")""")
            .mkString("\n")

      }
    }

  def regions_verification(implicit indent: Int): String =
    n_variadic_regions match {
      case 0 =>
        s"""if (regions.length != ${regions.length}) then throw new Exception(s"Expected ${regions.length} regions, got $${regions.length}")"""
      case 1 => {
        s"""if (regions.length < ${regions.length - 1}) then throw new Exception(s"Expected at least ${regions.length - 1} regions, got $${regions.length}")"""
      }
      case _: Int =>
        throw new NotImplementedError("Multivariadic regions not implemented")
    }

  def successors_verification(implicit indent: Int): String =
    n_variadic_successors match {
      case 0 =>
        s"""if (successors.length != ${successors.length}) then throw new Exception(s"Expected ${successors.length} successors, got $${successors.length}")"""
      case 1 => {
        s"""if (successors.length < ${successors.length - 1}) then throw new Exception(s"Expected at least ${successors.length - 1} successors, got $${successors.length}")"""
      }
      case _: Int =>
        throw new NotImplementedError("Multivariadic regions not implemented")
    }

  def constructs_verification(implicit indent: Int): String = s"""
    ${operands_verification(indent + 1)}
    ${results_verification(indent + 1)}
    ${regions_verification(indent + 1)}
    ${successors_verification(indent + 1)}
"""

  def irdl_verification(implicit indent: Int): String = s"""
  override def custom_verify(): Unit = 
    val verification_context = new ConstraintContext()
    ${constructs_verification(indent + 1)}
    ${constraints_verification(indent + 1)}
"""

  def print(implicit indent: Int): String = s"""
object $className extends OperationObject {
  override def name = "$name"
  override def factory = $className.apply
  ${assembly_format.map(f => f"val replace_by_parse = \"$f\"").getOrElse("")}
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

${assembly_format.map(f => f"val replace_by_print = \"$f\"").getOrElse("")}

${helpers(indent + 1)}
${accessors(indent + 1)}
${print_constr_defs(indent + 1)}
${irdl_verification(indent + 1)}

}
"""

}

/*≡≡=---=≡≡≡≡≡=---=≡≡*\
||   ATTRIBUTE DEF   ||
\*≡==----=≡≡≡=----==≡*/

case class AttributeDef(
    val name: String,
    val className: String,
    val parameters: Seq[OperandDef] = Seq(),
    val typee: Int = 0
) {

  def print(indent: Int): String = s"""
object $className extends AttributeObject {
  override def name = "$name"
  override def factory = $className.apply
}

case class $className(override val parameters: Seq[Attribute]) extends ParametrizedAttribute(name = "$name", parameters = parameters) ${
      if typee != 0 then "with TypeAttribute" else ""
    } {
  override def custom_verify(): Unit = 
    if (parameters.length != ${parameters.length}) then throw new Exception(s"Expected ${parameters.length} parameters, got $${parameters.length}")
}
  """

}

/*≡==--==≡≡≡≡≡≡≡==--=≡≡*\
||    CODE GEN HOOK    ||
\*≡==---==≡≡≡≡≡==---==≡*/

/** A helper class that generates a dialect implementation from a given dialect
  * definition.
  *
  * @param dialect_def
  *   The dialect definition to generate the implementation from.
  */
class ScaIRDLDialect(final val dialect_def: DialectDef) {

  /** Generates the dialect implementation.
    *
    * @param arg
    *   The path to the file to write the generated dialect implementation to.
    *   If the path is "-", the implementation will be written to the standard
    *   output.
    */
  final def main(args: Array[String]): Unit = {

    val writer = args(0) match {
      case "-" => System.out
      case arg => {
        val file = File(arg)
        file.getParentFile().mkdirs();
        file.createNewFile();
        PrintStream(file)
      }
    }

    writer.write(dialect_def.print(0).getBytes())
    writer.flush()
    writer.close()
  }

}

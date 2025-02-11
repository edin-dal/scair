package test.codegen

import scair.ir.Attribute
import scair.ir.Operation
import test.mirrored.*

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

/*≡≡=-=≡≡≡≡=-=≡≡*\
||  CONTAINERS  ||
\*≡=--==≡≡==--=≡*/

abstract class OpInput {}

// TODO: Add support for optionals AFTER variadic support is laid out
// It really just adds cognitive noise otherwise IMO. The broader structure and logic is exactly the same.
// (An Optional structurally is just a Variadic capped at one.)
enum Variadicity {
  case Single, Variadic
}

case class OperandDef(
    val id: String,
    val typeString: String,
    val variadicity: Variadicity = Variadicity.Single
) extends OpInput {}

case class ResultDef(
    val id: String,
    val typeString: String,
    val variadicity: Variadicity = Variadicity.Single
) extends OpInput {}

case class RegionDef(
    val id: String,
    val variadicity: Variadicity = Variadicity.Single
) extends OpInput {}

case class SuccessorDef(
    val id: String,
    val variadicity: Variadicity = Variadicity.Single
) extends OpInput {}

case class OpPropertyDef(
    val id: String,
    val typeString: String
) extends OpInput {}

case class OpAttributeDef(
    val id: String,
    val typeString: String
) extends OpInput {}

/*≡≡=---=≡≡≡≡≡≡≡≡=---=≡≡*\
||   MLIROP CONTAINER   ||
\*≡==----=≡≡≡≡≡≡=----==≡*/

case class MLIROpDef(
    val opDefs: Seq[OperationDef]
) {

  def getImports: String = {
    opDefs.map(x => x.getImports).flatten.distinct.mkString("\n")
  }

  def print: String = s"""
import scair.ir.Value
$getImports
${opDefs.map(_.print).mkString}
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
    val properties: Seq[OpPropertyDef] = Seq(),
    val attributes: Seq[OpAttributeDef] = Seq(),
    val assembly_format: Option[String] = None
) {

  /*≡≡=--=≡≡≡≡=---=≡≡*\
  ||      UTILS      ||
  \*≡==---==≡==---==≡*/

  def getImports: Seq[String] = {
    operands.map(x => s"import ${x.typeString}") ++
      results.map(x => s"import ${x.typeString}") ++
      properties.map(x => s"import ${x.typeString}") ++
      attributes.map(x => s"import ${x.typeString}")
  }

  /*≡≡=---=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=---=≡≡*\
  ||  UTILS FOR UNVERIFIED STATE CODEGEN  ||
  \*≡==-----====≡≡≡≡≡≡≡≡≡≡≡≡≡≡====-----==≡*/

  def getADTOpOperands: String =
    operands.map(x => s"op.${x.id}").mkString(", ")

  def getADTOpResults: String =
    results.map(x => s"op.${x.id}").mkString(", ")

  def getADTOpRegions: String =
    regions.map(x => s"op.${x.id}").mkString(", ")

  def getADTOpSuccessors: String =
    successors.map(x => s"op.${x.id}").mkString(", ")

  def getADTOpProperties: String =
    properties.map(x => s"(op.${x.id}, op.${x.id}.typ)").mkString(", ")

  def getADTOpAttributes: String =
    attributes.map(x => s"(op.${x.id}, op.${x.id}.typ)").mkString(", ")

  def getUnverifiedConstructor: String = {
    Seq(
      s"name = \"$name\"",
      if (operands.nonEmpty) s"operands = ListType(${getADTOpOperands})"
      else "",
      if (successors.nonEmpty) s"successors = ListType(${getADTOpSuccessors})"
      else "",
      if (regions.nonEmpty) s"regions = ListType(${getADTOpRegions})"
      else "",
      if (properties.nonEmpty)
        s"dictionaryProperties = DictType(${getADTOpProperties})"
      else "",
      if (attributes.nonEmpty)
        s"dictionaryAttributes = DictType(${getADTOpAttributes})"
      else ""
    ).filter(_ != "").mkString(",\n      ")
  }

  def getResultReassign: String = {
    if (results.nonEmpty) s"ListType(${getADTOpResults})"
    else ""
  }

  def unverifyGen: String = s"""
  def unverify(op: $className): RegisteredOp[$className] = {
    val op1 = RegisteredOp[$className](
      $getUnverifiedConstructor
    )

    op1.results.clear()
    op1.results.addAll(${getResultReassign})
    op1
  }
  """

  /*≡≡=---=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=---=≡≡*\
  ||  UTILS FOR VERIFIED STATE CODEGEN  ||
  \*≡==-----====≡≡≡≡≡≡≡≡≡≡≡≡====-----==≡*/

  def lenVerHelper(name: String, list: Seq[OpInput]): String = {
    if (!list.isEmpty)
      s"if (op.$name.length != ${list.length}) then throw new Exception(s\"Expected ${list.length} $name, got $${op.$name.length}\")\n    "
    else ""
  }

  def lengthVerification: String = {
    lenVerHelper("operands", operands) ++
      lenVerHelper("results", results) ++
      lenVerHelper("regions", regions) ++
      lenVerHelper("successors", successors) ++
      lenVerHelper("properties", properties)
  }

  def baseAttrVerOperands: Seq[String] = {
    operands.zipWithIndex
      .map((x, y) =>
        s"BaseAttr[${x.typeString}].verify(op.operands($y).typ, new ConstraintContext())"
      )
  }

  def baseAttrVerResults: Seq[String] = {
    results.zipWithIndex
      .map((x, y) =>
        s"BaseAttr[${x.typeString}].verify(op.results($y).typ, new ConstraintContext())"
      )
  }

  def baseAttrVerProperties: Seq[String] = {
    properties.zipWithIndex
      .map((x, y) =>
        s"BaseAttr[${x.typeString}].verify(op.properties(${x.id}), new ConstraintContext())"
      )
  }

  def baseAttrVerification: String = {
    (baseAttrVerOperands ++ baseAttrVerResults ++ baseAttrVerProperties)
      .mkString("\n    ")
  }

  def getREGOpOperands: Seq[String] =
    (operands zip (0 to operands.length).toList)
      .map((x, y) =>
        s"${x.id} = op.operands($y).asInstanceOf[Value[${x.typeString}]]"
      )

  def getREGOpResults: Seq[String] =
    (results zip (0 to results.length).toList)
      .map((x, y) =>
        s"${x.id} = op.results($y).asInstanceOf[Value[${x.typeString}]]"
      )

  def getREGOpRegions: Seq[String] =
    (regions zip (0 to regions.length).toList)
      .map((x, y) => s"${x.id} = op.regions($y)")

  def getREGOpSuccessors: Seq[String] =
    (successors zip (0 to successors.length).toList)
      .map((x, y) => s"${x.id} = op.successors($y)")

  def getREGOpProperties: Seq[String] =
    properties
      .map(x =>
        s"${x.id} = Property(op.properties(${x.id}).asInstanceOf[Value[${x.typeString}]])"
      )

  def getREGOpAttributes: Seq[String] =
    attributes
      .map(x =>
        s"${x.id} = Attr(op.attributes(${x.id}).asInstanceOf[Value[${x.typeString}]])"
      )

  def getVerifiedConstructor: String = {
    (getREGOpOperands ++
      getREGOpResults ++
      getREGOpRegions ++
      getREGOpSuccessors ++
      getREGOpProperties ++
      getREGOpAttributes).mkString(",\n      ")
  }

  def verifyGen: String = s"""
  def verify(op: RegisteredOp[_]): $className = {
    $lengthVerification
    $baseAttrVerification

    $className(
      $getVerifiedConstructor
    )
  }
  """

  def print: String = s"""

val mlirOp$className = new MLIRRealm[$className] {
  $unverifyGen
  $verifyGen
}
    
"""

}

/*
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

  def single_operand_accessor(name: String, index: String) =
    s"  def ${name}: Value[Attribute] = operands($index)\n" +
      s"  def ${name}_=(new_operand: Value[Attribute]): Unit = {operands($index) = new_operand}\n"

  def segmented_single_operand_accessor(name: String, index: String) =
    single_operand_accessor(
      name,
      s"operandSegmentSizes.slice(0, $index).fold(0)(_ + _)"
    )

  def variadic_operand_accessor(name: String, from: String, to: String) =
    s"""  def ${name}: Seq[Value[Attribute]] = {
      val from = $from
      val to = $to
      operands.slice(from, to).toSeq
  }
  def ${name}_=(new_operands: Seq[Value[Attribute]]): Unit = {
    val from = $from
    val to = $to
    val diff = new_operands.length - (to - from)
    for (operand, i) <- (new_operands ++ operands.slice(to, operands.length)).zipWithIndex do
      operands(from + i) = operand
    if (diff < 0)
      operands.trimEnd(-diff)
  }\n\n"""

  def segmented_variadic_operand_accessor(name: String, index: String) =
    variadic_operand_accessor(
      name,
      s"operandSegmentSizes.slice(0, $index).fold(0)(_ + _)",
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

  // TODO: We *probably* really don't want this setter or at least that way.
  // That's more of a generic framework question though and is orthogonal to this codegen thing
  // hence just a TODO for later.
  def single_result_accessor(name: String, index: String) =
    s"  def ${name}: Value[Attribute] = results($index)\n" +
      s"  def ${name}_=(new_result: Value[Attribute]): Unit = {results($index) = new_result}\n"

  def segmented_single_result_accessor(name: String, index: String) =
    single_result_accessor(
      name,
      s"resultSegmentSizes.slice(0, $index).fold(0)(_ + _)"
    )

  // TODO: We *probably* really don't want this setter or at least that way.
  // That's more of a generic framework question though and is orthogonal to this codegen thing
  // hence just a TODO for later.
  def variadic_result_accessor(name: String, from: String, to: String) =
    s"""  def ${name}: Seq[Value[Attribute]] = {
      val from = $from
      val to = $to
      results.slice(from, to).toSeq
  }
  def ${name}_=(new_results: Seq[Value[Attribute]]): Unit = {
    val from = $from
    val to = $to
    val diff = new_results.length - (to - from)
    for (new_results, i) <- (new_results ++ results.slice(to, results.length)).zipWithIndex do
      results(from + i) = new_results
    if (diff < 0)
      results.trimEnd(-diff)
  }\n\n"""

  def segmented_variadic_result_accessor(name: String, index: String) =
    variadic_result_accessor(
      name,
      s"resultSegmentSizes.slice(0, $index).fold(0)(_ + _)",
      s"from + resultSegmentSizes($index)"
    )

  def results_accessors(implicit indent: Int): Seq[String] = {
    n_variadic_results match {
      case 0 =>
        for ((odef, i) <- results.zipWithIndex)
          yield single_result_accessor(odef.id, i.toString)
      case 1 => {
        val variadic_index =
          results.indexWhere(_.variadicity != Variadicity.Single);
        ((
          for (odef, i) <- results.slice(0, variadic_index).zipWithIndex
          yield single_result_accessor(odef.id, i.toString)
        )
          :+
            (variadic_result_accessor(
              results(variadic_index).id,
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
            yield single_result_accessor(
              odef.id,
              s"results.length - ${results.length - i}"
            ))

      }
      case _: Int => {
        for ((odef, i) <- results.zipWithIndex)
          yield odef.variadicity match {
            case Variadicity.Single =>
              segmented_single_result_accessor(odef.id, i.toString)
            case Variadicity.Variadic =>
              segmented_variadic_result_accessor(odef.id, i.toString)
          }
      }
    }
  }

  def single_region_accessor(name: String, index: String) =
    s"  def ${name}: Region = regions($index)\n" +
      s"  def ${name}_=(new_region: Region): Unit = {regions($index) = new_region}\n"

  def variadic_region_accessor(name: String, from: String, to: String) =
    s"""  def ${name}: Seq[Region] = {
      val from = $from
      val to = $to
      regions.slice(from, to).toSeq
  }
  def ${name}_=(new_regions: Seq[Region]): Unit = {
    val from = $from
    val to = $to
    val diff = new_regions.length - (to - from)
    for (region, i) <- (new_regions ++ regions.slice(to, regions.length)).zipWithIndex do
      regions(from + i) = region
    if (diff < 0)
      regions.trimEnd(-diff)
  }\n\n"""

  def regions_accessors(implicit indent: Int): Seq[String] = {
    n_variadic_regions match {
      case 0 =>
        for ((odef, i) <- regions.zipWithIndex)
          yield single_region_accessor(odef.id, i.toString)
      case 1 => {
        val variadic_index =
          regions.indexWhere(_.variadicity != Variadicity.Single);
        ((
          for (odef, i) <- regions.slice(0, variadic_index).zipWithIndex
          yield single_region_accessor(odef.id, i.toString)
        )
          :+
            (variadic_region_accessor(
              regions(variadic_index).id,
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
            yield single_region_accessor(
              odef.id,
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

  def single_successor_accessor(name: String, index: String) =
    s"  def ${name}: Block = successors($index)\n" +
      s"  def ${name}_=(new_successor: Block): Unit = {successors($index) = new_successor}\n"

  def variadic_successor_accessor(name: String, from: String, to: String) =
    s"""  def ${name}: Seq[Block] = {
      val from = $from
      val to = $to
      successors.slice(from, to).toSeq
  }
  def ${name}_=(new_successors: Seq[Block]): Unit = {
    val from = $from
    val to = $to
    val diff = new_successors.length - (to - from)
    for (successor, i) <- (new_successors ++ successors.slice(to, successors.length)).zipWithIndex do
      successors(from + i) = successor
    if (diff < 0)
      successors.trimEnd(-diff)
  }\n\n"""

  def successors_accessors(implicit indent: Int): Seq[String] = {
    n_variadic_successors match {
      case 0 =>
        for ((odef, i) <- successors.zipWithIndex)
          yield single_successor_accessor(odef.id, i.toString)
      case 1 => {
        val variadic_index =
          successors.indexWhere(_.variadicity != Variadicity.Single);
        ((
          for (odef, i) <- successors.slice(0, variadic_index).zipWithIndex
          yield single_successor_accessor(odef.id, i.toString)
        )
          :+
            (variadic_successor_accessor(
              successors(variadic_index).id,
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
            yield single_successor_accessor(
              odef.id,
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
    results_types: ListType[Attribute] = ListType(),
    override val regions: ListType[Region] = ListType(),
    override val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends RegisteredOperation(name = "$name", operands, successors, results_types, regions, dictionaryProperties, dictionaryAttributes) {

${assembly_format.map(f => f"val replace_by_print = \"$f\"").getOrElse("")}

${helpers(indent + 1)}
${accessors(indent + 1)}
${print_constr_defs(indent + 1)}
${irdl_verification(indent + 1)}

}
"""
 */

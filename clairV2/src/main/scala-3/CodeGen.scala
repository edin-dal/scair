package scair.clairV2.codegen

import java.io.File
import java.io.PrintStream
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

  def print: String = s"""package ${opDefs(0).packageName}

import scair.ir.*
import scair.ir.ValueConversions.{resToVal, valToRes}
import scair.scairdl.constraints.{BaseAttr, ConstraintContext}
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
    val packageName: String,
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
    properties.map(x => s"(\"${x.id}\", op.${x.id}.typ)").mkString(", ")

  def getADTOpAttributes: String =
    attributes.map(x => s"(\"${x.id}\", op.${x.id}.typ)").mkString(", ")

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
    ).filter(_ != "").mkString(",\n        ")
  }

  def getResultReassign: String = {
    if (results.nonEmpty) s"ListType(${getADTOpResults})"
    else ""
  }

  def unverifyGen: String = s"""
    def unverify(op: $className): UnverifiedOp[$className] = {
      val op1 = UnverifiedOp[$className](
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

  // length verification

  def lenVerHelper(name: String, list: Seq[OpInput]): String = {
    if (!list.isEmpty)
      s"if (op.$name.size != ${list.length}) then throw new Exception(s\"Expected ${list.length} $name, got $${op.$name.size}\")\n      "
    else ""
  }

  def lengthVerification: String = {
    lenVerHelper("operands", operands) ++
      lenVerHelper("results", results) ++
      lenVerHelper("regions", regions) ++
      lenVerHelper("successors", successors) ++
      lenVerHelper("dictionaryProperties", properties) ++
      lenVerHelper("dictionaryAttributes", attributes)
  }

  // have to verify that the specific properties and attributes exist
  def propAndAttrExist: String = {

    val ver =
      (properties
        .map(x => s"op.dictionaryProperties.contains(\"${x.id}\")") ++
        attributes.map(x => s"op.dictionaryAttributes.contains(\"${x.id}\")"))

    if !(ver.isEmpty) then s"""
      if !(${ver.mkString(" &&\n      ")}) 
      then throw new Exception("Expected specific properties and attributes")  
""" else ""
  }

  // base attribute verification / type verification

  def baseAttrVerOperands: Seq[String] = {
    operands.zipWithIndex
      .map((x, y) =>
        s"BaseAttr[${x.typeString}]().verify(op.operands($y).typ, new ConstraintContext())"
      )
  }

  def baseAttrVerResults: Seq[String] = {
    results.zipWithIndex
      .map((x, y) =>
        s"BaseAttr[${x.typeString}]().verify(op.results($y).typ, new ConstraintContext())"
      )
  }

  def baseAttrVerProperties: Seq[String] = {
    properties.zipWithIndex
      .map((x, y) =>
        s"BaseAttr[${x.typeString}]().verify(op.dictionaryProperties(\"${x.id}\"), new ConstraintContext())"
      )
  }

  def baseAttrVerAttributes: Seq[String] = {
    attributes.zipWithIndex
      .map((x, y) =>
        s"BaseAttr[${x.typeString}]().verify(op.dictionaryAttributes(\"${x.id}\"), new ConstraintContext())"
      )
  }

  def baseAttrVerification: String = {
    (baseAttrVerOperands ++ baseAttrVerResults ++ baseAttrVerProperties ++ baseAttrVerAttributes)
      .mkString("\n      ")
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
        s"${x.id} = Property(op.dictionaryProperties(\"${x.id}\").asInstanceOf[${x.typeString}])"
      )

  def getREGOpAttributes: Seq[String] =
    attributes
      .map(x =>
        s"${x.id} = Attr(op.dictionaryAttributes(\"${x.id}\").asInstanceOf[${x.typeString}])"
      )

  def getVerifiedConstructor: String = {
    (getREGOpOperands ++
      getREGOpResults ++
      getREGOpRegions ++
      getREGOpSuccessors ++
      getREGOpProperties ++
      getREGOpAttributes).mkString(",\n        ")
  }

  def verifyGen: String = s"""
    def verify(op: UnverifiedOp[$className]): $className = {

      $lengthVerification
      $propAndAttrExist
      $baseAttrVerification

      $className(
        $getVerifiedConstructor
      )
    }
"""

  /*≡≡=---=≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡=---=≡≡*\
  ||    CONSTRUCTOR FOR UNVERIFIED OP    ||
  \*≡==-----====≡≡≡≡≡≡≡≡≡≡≡≡≡====-----==≡*/

  def unverifiedConstructorGen: String = s"""

    def constructUnverifiedOp(
        operands: ListType[Value[Attribute]] = ListType(),
        successors: ListType[Block] = ListType(),
        results_types: ListType[Attribute] = ListType(),
        regions: ListType[Region] = ListType(),
        dictionaryProperties: DictType[String, Attribute] = DictType.empty[String, Attribute],
        dictionaryAttributes: DictType[String, Attribute] = DictType.empty[String, Attribute]
    ): UnverifiedOp[$className] = {
      UnverifiedOp[$className](
        name = "$name",
        operands = operands,
        successors = successors,
        results_types = results_types,
        regions = regions,
        dictionaryProperties = dictionaryProperties,
        dictionaryAttributes = dictionaryAttributes
      )
    }
"""

  def print: String = s"""

object ${className}Helper extends ADTCompanion {
  val getMLIRRealm: MLIRRealm[$className] = new MLIRRealm[$className] {
    $unverifiedConstructorGen
    $unverifyGen
    $verifyGen
  }
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
class MLIRRealmize(final val mlirOpDef: MLIROpDef) {

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

    writer.write(mlirOpDef.print.getBytes())
    writer.flush()
    writer.close()
  }

}

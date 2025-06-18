package scair.clair.codegen

import scala.quoted.*
import scala.reflect.*

// ░█████╗░ ██╗░░░░░ ░█████╗░ ██╗ ██████╗░ ██╗░░░██╗ ██████╗░
// ██╔══██╗ ██║░░░░░ ██╔══██╗ ██║ ██╔══██╗ ██║░░░██║ ╚════██╗
// ██║░░╚═╝ ██║░░░░░ ███████║ ██║ ██████╔╝ ╚██╗░██╔╝ ░░███╔═╝
// ██║░░██╗ ██║░░░░░ ██╔══██║ ██║ ██╔══██╗ ░╚████╔╝░ ██╔══╝░░
// ╚█████╔╝ ███████╗ ██║░░██║ ██║ ██║░░██║ ░░╚██╔╝░░ ███████╗
// ░╚════╝░ ╚══════╝ ╚═╝░░╚═╝ ╚═╝ ╚═╝░░╚═╝ ░░░╚═╝░░░ ╚══════╝

// ░█████╗░ ░█████╗░ ██████╗░ ███████╗ ░██████╗░ ███████╗ ███╗░░██╗
// ██╔══██╗ ██╔══██╗ ██╔══██╗ ██╔════╝ ██╔════╝░ ██╔════╝ ████╗░██║
// ██║░░╚═╝ ██║░░██║ ██║░░██║ █████╗░░ ██║░░██╗░ █████╗░░ ██╔██╗██║
// ██║░░██╗ ██║░░██║ ██║░░██║ ██╔══╝░░ ██║░░╚██╗ ██╔══╝░░ ██║╚████║
// ╚█████╔╝ ╚█████╔╝ ██████╔╝ ███████╗ ╚██████╔╝ ███████╗ ██║░╚███║
// ░╚════╝░ ░╚════╝░ ╚═════╝░ ╚══════╝ ░╚═════╝░ ╚══════╝ ╚═╝░░╚══╝

/*≡≡=-=≡≡≡≡=-=≡≡*\
||  CONTAINERS  ||
\*≡=--==≡≡==--=≡*/

sealed trait OpInputDef {
  def name: String
}

sealed trait MayVariadicOpInputDef extends OpInputDef {
  def variadicity: Variadicity
}

// TODO: Add support for optionals AFTER variadic support is laid out
// It really just adds cognitive noise otherwise IMO. The broader structure and logic is exactly the same.
// (An Optional structurally is just a Variadic capped at one.)
enum Variadicity {
  case Single, Variadic, Optional
}

case class OperandDef(
    override val name: String,
    val tpe: Type[?],
    override val variadicity: Variadicity = Variadicity.Single
) extends OpInputDef
    with MayVariadicOpInputDef {}

case class ResultDef(
    override val name: String,
    val tpe: Type[?],
    override val variadicity: Variadicity = Variadicity.Single
) extends OpInputDef
    with MayVariadicOpInputDef {}

case class RegionDef(
    override val name: String,
    override val variadicity: Variadicity = Variadicity.Single
) extends OpInputDef
    with MayVariadicOpInputDef {}

case class SuccessorDef(
    override val name: String,
    override val variadicity: Variadicity = Variadicity.Single
) extends OpInputDef
    with MayVariadicOpInputDef {}

case class OpPropertyDef(
    override val name: String,
    val tpe: Type[?],
    val optional: Boolean = false
) extends OpInputDef {}

case class AttributeParamDef(
    val name: String,
    val tpe: Type[?]
) {}

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
    val assembly_format: Option[String] = None
) {

  def allDefsWithIndex =
    operands.zipWithIndex ++ results.zipWithIndex ++ regions.zipWithIndex ++ successors.zipWithIndex ++ properties.zipWithIndex

  def hasMultiVariadicOperands =
    operands.count(_.variadicity == Variadicity.Variadic) > 1

  def hasMultiVariadicResults =
    results.count(_.variadicity == Variadicity.Variadic) > 1

  def hasMultiVariadicRegions =
    regions.count(_.variadicity == Variadicity.Variadic) > 1

  def hasMultiVariadicSuccessors =
    successors.count(_.variadicity == Variadicity.Variadic) > 1

}

/*≡≡=---=≡≡≡≡≡=---=≡≡*\
||   ATTRIBUTE DEF   ||
\*≡==----=≡≡≡=----==≡*/

case class AttributeDef(
    val name: String,
    val attributes: Seq[AttributeParamDef] = Seq()
) {

  def allDefsWithIndex =
    attributes.zipWithIndex

}

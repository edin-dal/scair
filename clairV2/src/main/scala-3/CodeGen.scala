package scair.clairV2.codegen

import java.io.File
import java.io.PrintStream
import scala.reflect.*
import scair.ir._
import scala.quoted._

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

sealed trait OpInputDef(val name: String) {}

sealed trait MayVariadicOpInputDef(val variadicity: Variadicity)
    extends OpInputDef

// TODO: Add support for optionals AFTER variadic support is laid out
// It really just adds cognitive noise otherwise IMO. The broader structure and logic is exactly the same.
// (An Optional structurally is just a Variadic capped at one.)
enum Variadicity {
  case Single, Variadic
}

type DefinedInputOf[T <: OpInputDef, A <: Attribute] = T match {
  case OperandDef    => Operand[A]
  case ResultDef     => Result[A]
  case RegionDef     => Region
  case SuccessorDef  => Successor
  case OpPropertyDef => Property[A]
}

type DefinedInput[T <: OpInputDef] = DefinedInputOf[T, Attribute]

case class OperandDef(
    override val name: String,
    val tpe: Type[_],
    override val variadicity: Variadicity = Variadicity.Single
) extends OpInputDef(name)
    with MayVariadicOpInputDef(variadicity) {}

case class ResultDef(
    override val name: String,
    val tpe: Type[_],
    override val variadicity: Variadicity = Variadicity.Single
) extends OpInputDef(name)
    with MayVariadicOpInputDef(variadicity) {}

case class RegionDef(
    override val name: String,
    override val variadicity: Variadicity = Variadicity.Single
) extends OpInputDef(name)
    with MayVariadicOpInputDef(variadicity) {}

case class SuccessorDef(
    override val name: String,
    override val variadicity: Variadicity = Variadicity.Single
) extends OpInputDef(name)
    with MayVariadicOpInputDef(variadicity) {}

case class OpPropertyDef(
    override val name: String,
    val tpe: Type[_]
) extends OpInputDef(name) {}

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

}

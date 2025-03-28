package scair.clairV2.codegen

import java.io.File
import java.io.PrintStream
import scala.reflect.*
import scair.ir._
import scala.quoted._
import fastparse._
import fastparse.ScalaWhitespace.*

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

case class Assemblyformat(
    format: String,
    operands: Seq[String], //  ["$lhs", "$rhs"]
    types: Seq[String], //  ["type($lhs)", "type($rhs)"]
    results: Seq[String] //  ["type($result)"]
)

trait FormatDirective

case class LiteralDirective(literal: String) extends FormatDirective
case class ResultTypeDirective(name: String = "result") extends FormatDirective
case class OperandDirective(name: String) extends FormatDirective
case class TypeDirective(inner: OperandDirective) extends FormatDirective

object NewParser {

  def resultTypeDirective[$: P]: P[ResultTypeDirective] =
    P("type($result)").map(_ => ResultTypeDirective())

  def operandDirective[$: P]: P[OperandDirective] =
    P("$" ~ CharsWhileIn("a-zA-Z0-9_").!)
      .filter(_ != "result")
      .map(OperandDirective)

  def typeDirective[$: P]: P[TypeDirective] =
    P("type(" ~ (operandDirective) ~ ")").map(TypeDirective)

  def literalDirective[$: P]: P[LiteralDirective] =
    P("`" ~ CharsWhile(_ != '`').! ~ "`").map(LiteralDirective)

  def formatDirective[$: P]: P[FormatDirective] =
    P(typeDirective | operandDirective | literalDirective | resultTypeDirective)

  def assemblyFormat[$: P]: P[Seq[FormatDirective]] =
    P(formatDirective.rep(1))

  def parseFormat(input: String): Parsed[Seq[FormatDirective]] =
    parse(input, assemblyFormat(_))

}

def Parseassemblyformat(format: String): Seq[FormatDirective] = {
  NewParser.parseFormat(format) match {
    case Parsed.Success(parsedFormat, _) => parsedFormat
    case Parsed.Failure(_, index, extra) =>
      throw new Exception(
        s"Parsing failed at index $index}"
      )
  }
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
    val assembly_format: Option[Seq[FormatDirective]] = None
) {

  def allDefsWithIndex =
    operands.zipWithIndex ++ results.zipWithIndex ++ regions.zipWithIndex ++ successors.zipWithIndex ++ properties.zipWithIndex

}

package scair.dialects.math

import fastparse.*
import scair.AttrParser.whitespace
import scair.Parser
import scair.Parser.orElse
import scair.clair.macros.DerivedOperation
import scair.clair.macros.summonDialect
import scair.dialects.arith.FastMathFlags
import scair.dialects.arith.FastMathFlagsAttr
import scair.dialects.builtin.*
import scair.ir.*

//
// ███╗░░░███╗ ░█████╗░ ████████╗ ██╗░░██╗
// ████╗░████║ ██╔══██╗ ╚══██╔══╝ ██║░░██║
// ██╔████╔██║ ███████║ ░░░██║░░░ ███████║
// ██║╚██╔╝██║ ██╔══██║ ░░░██║░░░ ██╔══██║
// ██║░╚═╝░██║ ██║░░██║ ░░░██║░░░ ██║░░██║
// ╚═╝░░░░░╚═╝ ╚═╝░░╚═╝ ░░░╚═╝░░░ ╚═╝░░╚═╝
//

////////////////
// OPERATIONS //
////////////////

// ==--------== //
//   AbsfOp   //
// ==--------== //

object AbsfOp:
  def name: String = "math.absf"

  def parse[$: P](
      parser: Parser,
      resNames: Seq[String]
  ): P[Operation] =
    P(
      "" ~ Parser.ValueUse ~ parser.Attribute.orElse(
        FastMathFlagsAttr(FastMathFlags.none)
      ) ~ ":" ~ parser.Type
    ).map { case (operandName, flags, type_) =>
      parser.generateOperation(
        opName = name,
        operandsNames = Seq(operandName),
        operandsTypes = Seq(type_),
        resultsNames = resNames,
        resultsTypes = Seq(type_),
        properties = Map("fastmath" -> flags)
      )
    }

case class AbsfOp(
    fastmath: FastMathFlagsAttr,
    operand: Operand[FloatType],
    result: Result[FloatType]
) extends DerivedOperation["math.absf", AbsfOp]
    with NoMemoryEffect

// ==--------== //
//   FPowIOp   //
// ==--------== //

object FPowIOp:
  def name: String = "math.fpowi"

  def parse[$: P](
      parser: Parser,
      resNames: Seq[String]
  ): P[Operation] =
    P(
      Parser.ValueUse ~ "," ~ Parser.ValueUse ~ parser.Attribute.orElse(
        FastMathFlagsAttr(FastMathFlags.none)
      ) ~ ":" ~ parser.Type ~ "," ~ parser.Type
    ).map {
      case (
            operand1Name,
            operand2Name,
            flags,
            operand1Type,
            operand2Type
          ) =>
        parser.generateOperation(
          opName = name,
          operandsNames = Seq(operand1Name, operand2Name),
          operandsTypes = Seq(operand1Type, operand2Type),
          resultsNames = resNames,
          resultsTypes = Seq(operand1Type),
          properties = Map("fastmath" -> flags)
        )
    }

case class FPowIOp(
    lhs: Operand[FloatType],
    rhs: Operand[IntegerType],
    fastmath: FastMathFlagsAttr,
    result: Result[FloatType]
) extends DerivedOperation["math.fpowi", FPowIOp]
    with NoMemoryEffect

/////////////
// DIALECT //
/////////////

val MathDialect: Dialect =
  summonDialect[EmptyTuple, (AbsfOp, FPowIOp)]

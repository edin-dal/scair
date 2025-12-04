package scair.dialects.math

import fastparse.*
import scair.AttrParser.whitespace
import scair.Parser
import scair.Parser.orElse
import scair.clair.macros.*
import scair.clair.macros.DerivedOperation
import scair.clair.macros.DerivedOperationCompanion
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
  ): P[AbsfOp] =
    P(
      "" ~ Parser.ValueUse
        .flatMap(operandName =>
          (parser.Attribute.orElse(
            FastMathFlagsAttr(FastMathFlags.none)
          ) ~ ":" ~ parser.Type.flatMap(tpe =>
            parser.currentScope.useValue(operandName, tpe) ~
              parser.currentScope.defineResult(resNames.head, tpe)
          ))
        )
        .flatMap { case (flags, operandAndResult) =>
          val (operand, result) = operandAndResult
          summon[DerivedOperationCompanion[AbsfOp]]
            .apply(
              operands = Seq(operand),
              results = Seq(result),
              properties = Map("fastmath" -> flags)
            )
            .structured match
            case Right(op: AbsfOp) => Pass(op)
            case Left(err)         => Fail(err)

        }
    )

case class AbsfOp(
    fastmath: FastMathFlagsAttr,
    operand: Operand[FloatType],
    result: Result[FloatType]
) extends DerivedOperation["math.absf", AbsfOp]
    with NoMemoryEffect derives DerivedOperationCompanion

// ==--------== //
//   FPowIOp   //
// ==--------== //

object FPowIOp:
  def name: String = "math.fpowi"

  def parse[$: P](
      parser: Parser,
      resNames: Seq[String]
  ): P[FPowIOp] =
    P(
      Parser.ValueUse.flatMap(lhsName =>
        ("," ~ Parser.ValueUse.flatMap(rhsName =>
          (parser.Attribute.orElse(
            FastMathFlagsAttr(FastMathFlags.none)
          ) ~ ":" ~ parser.Type.flatMap(lhsType =>
            parser.currentScope.useValue(lhsName, lhsType) ~
              parser.currentScope.defineResult(resNames.head, lhsType)
          )
            ~ "," ~ parser.Type.flatMap(
              parser.currentScope.useValue(rhsName, _)
            ))
            .flatMap {
              case (
                    flags,
                    lhsAndRes,
                    rhs
                  ) =>
                println(f"Attempting construction")
                val made = summon[DerivedOperationCompanion[FPowIOp]].apply(
                  operands = Seq(lhsAndRes._1, rhs),
                  results = Seq(lhsAndRes._2),
                  properties = Map("fastmath" -> flags)
                )
                println(f"Made: $made")
                made.structured match
                  case Right(op: FPowIOp) => Pass(op)
                  case Left(err)          => Fail(err)

            }
        ))
      )
    )

case class FPowIOp(
    lhs: Operand[FloatType],
    rhs: Operand[IntegerType],
    fastmath: FastMathFlagsAttr,
    result: Result[FloatType]
) extends DerivedOperation["math.fpowi", FPowIOp]
    with NoMemoryEffect derives DerivedOperationCompanion

/////////////
// DIALECT //
/////////////

val MathDialect: Dialect =
  summonDialect[EmptyTuple, (AbsfOp, FPowIOp)]

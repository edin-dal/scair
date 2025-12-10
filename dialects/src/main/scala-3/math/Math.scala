package scair.dialects.math

import fastparse.*
import scair.*
import scair.clair.macros.*
import scair.clair.macros.DerivedOperation
import scair.clair.macros.DerivedOperationCompanion
import scair.clair.macros.summonDialect
import scair.dialects.arith.FastMathFlags
import scair.dialects.arith.FastMathFlagsAttr
import scair.dialects.builtin.*
import scair.ir.*
import scair.parse.*

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
given OperationCustomParser[AbsfOp]:

  def parse[$: P](
      resNames: Seq[String]
  )(using p: Parser): P[AbsfOp] =
    P(
      OperandName.flatMap(operandName =>
          (AttributeP.orElse(
            FastMathFlagsAttr(FastMathFlags.none)
          ) ~ ":" ~ TypeP.flatMap(tpe =>
            operand(operandName, tpe) ~
              result(resNames.head, tpe)
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

given OperationCustomParser[FPowIOp]:

  def parse[$: P](
      resNames: Seq[String]
  )(using p: Parser): P[FPowIOp] =
    P(
      OperandName.flatMap(lhsName =>
        ("," ~ OperandName.flatMap(rhsName =>
          (AttributeP.orElse(
            FastMathFlagsAttr(FastMathFlags.none)
          ) ~ ":" ~ TypeP.flatMap(lhsType =>
            operand(lhsName, lhsType) ~
              result(resNames.head, lhsType)
          )
            ~ "," ~ TypeP.flatMap(
              operand(rhsName, _)
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

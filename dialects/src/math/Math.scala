package scair.dialects.math

import fastparse.*
import scair.*
import scair.clair.*
import scair.clair.DerivedOperation
import scair.clair.DerivedOperationCompanion
import scair.clair.summonDialect
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
      operandNameP.flatMap(operandName =>
        (attrOfOrP[FastMathFlagsAttr](
          FastMathFlags.none
        ) ~ ":" ~ typeOfP[FloatType].flatMap(tpe =>
          operandP(operandName, tpe) ~ resultP(resNames.head, tpe)
        ))
      ).map { case (flags, operandAndResult) =>
        val (operand, result) = operandAndResult
        AbsfOp(flags, operand, result)
      }
    )

case class AbsfOp(
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    operand: Operand[FloatType],
    result: Result[FloatType],
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
      operandNameP.flatMap(lhsName =>
        ("," ~ operandNameP.flatMap(rhsName =>
          (attrOfOrP[FastMathFlagsAttr](
            FastMathFlags.none
          ) ~ ":" ~ typeOfP[FloatType].flatMap(lhsType =>
            operandP(lhsName, lhsType) ~ resultP(resNames.head, lhsType)
          ) ~ "," ~ typeOfP[IntegerType].flatMap(
            operandP(rhsName, _)
          )).map { case (flags, lhsAndRes, rhs) =>
            val (lhs, res) = lhsAndRes
            FPowIOp(lhs, rhs, flags, res)

          }
        ))
      )
    )

case class FPowIOp(
    lhs: Operand[FloatType],
    rhs: Operand[IntegerType],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatType],
) extends DerivedOperation["math.fpowi", FPowIOp]
    with NoMemoryEffect derives DerivedOperationCompanion

/////////////
// DIALECT //
/////////////

val MathDialect: Dialect =
  summonDialect[EmptyTuple, (AbsfOp, FPowIOp)]

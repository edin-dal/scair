package scair.dialects.math

import fastparse.*
import scair.*
import scair.clair.*
import scair.clair.DerivedOperation
import scair.clair.OpDefs
import scair.clair.summonDialect
import scair.dialects.arith.*
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

// TODO: should constrain to floats, vectors or tensors thereof.
type FloatLike = FloatType | VectorType | ContainerType
// TODO: should constrain to signless integers or index, vectors or tensors thereof.
type IntegerOrIndexLike = AnyIntegerType | VectorType | ContainerType

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
) extends DerivedOperation["math.absf"]
    with Pure derives OpDefs

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
) extends DerivedOperation["math.fpowi"]
    with Pure derives OpDefs

case class AcosOp(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.acos"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class AcoshOp(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.acosh"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class AsinOp(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.asin"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class AsinhOp(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.asinh"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class AtanOp(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.atan"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class AtanhOp(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.atanh"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class CbrtOp(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.cbrt"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class CeilOp(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.ceil"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class CosOp(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.cos"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class CoshOp(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.cosh"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class ErfOp(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.erf"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class ErfcOp(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.erfc"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class ExpOp(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.exp"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class Exp2Op(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.exp2"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class ExpM1Op(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.expm1"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class FloorOp(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.floor"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class LogOp(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.log"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class Log10Op(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.log10"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class Log1pOp(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.log1p"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class Log2Op(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.log2"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class RoundOp(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.round"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class RoundEvenOp(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.roundeven"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class RsqrtOp(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.rsqrt"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class SinOp(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.sin"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class SinhOp(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.sinh"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class SqrtOp(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.sqrt"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class TanOp(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.tan"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class TanhOp(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.tanh"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class TruncOp(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.trunc"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class Atan2Op(
    lhs: Operand[FloatLike],
    rhs: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.atan2"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class CopySignOp(
    lhs: Operand[FloatLike],
    rhs: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.copysign"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class PowFOp(
    lhs: Operand[FloatLike],
    rhs: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.powf"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class FmaOp(
    a: Operand[FloatLike],
    b: Operand[FloatLike],
    c: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.fma"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class ClampFOp(
    value: Operand[FloatLike],
    min: Operand[FloatLike],
    max: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[FloatLike],
) extends DerivedOperation["math.clampf"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class AbsIOp(
    operand: Operand[IntegerOrIndexLike],
    result: Result[IntegerOrIndexLike],
) extends DerivedOperation["math.absi"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class CountLeadingZerosOp(
    operand: Operand[IntegerOrIndexLike],
    result: Result[IntegerOrIndexLike],
) extends DerivedOperation["math.ctlz"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class CountTrailingZerosOp(
    operand: Operand[IntegerOrIndexLike],
    result: Result[IntegerOrIndexLike],
) extends DerivedOperation["math.cttz"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class CtPopOp(
    operand: Operand[IntegerOrIndexLike],
    result: Result[IntegerOrIndexLike],
) extends DerivedOperation["math.ctpop"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class IPowIOp(
    lhs: Operand[IntegerOrIndexLike],
    rhs: Operand[IntegerOrIndexLike],
    result: Result[IntegerOrIndexLike],
) extends DerivedOperation["math.ipowi"]
    with SameOperandsAndResultTypes
    with Pure derives OpDefs

case class IsFiniteOp(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[BoolLike],
) extends DerivedOperation["math.isfinite"]
    with Pure derives OpDefs

case class IsInfOp(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[BoolLike],
) extends DerivedOperation["math.isinf"]
    with Pure derives OpDefs

case class IsNaNOp(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[BoolLike],
) extends DerivedOperation["math.isnan"]
    with Pure derives OpDefs

case class IsNormalOp(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    result: Result[BoolLike],
) extends DerivedOperation["math.isnormal"]
    with Pure derives OpDefs

case class SincosOp(
    operand: Operand[FloatLike],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
    sin: Result[FloatLike],
    cos: Result[FloatLike],
) extends DerivedOperation["math.sincos"]
    with SameOperandsAndResultShape
    with AllTypesMatch(operand.typ, sin.typ, cos.typ)
    with Pure derives OpDefs

/////////////
// DIALECT //
/////////////

val MathDialect: Dialect =
  summonDialect[
    EmptyTuple,
    (
        AbsfOp,
        FPowIOp,
        AcosOp,
        AcoshOp,
        AsinOp,
        AsinhOp,
        AtanOp,
        AtanhOp,
        CbrtOp,
        CeilOp,
        CosOp,
        CoshOp,
        ErfOp,
        ErfcOp,
        ExpOp,
        Exp2Op,
        ExpM1Op,
        FloorOp,
        LogOp,
        Log10Op,
        Log1pOp,
        Log2Op,
        RoundOp,
        RoundEvenOp,
        RsqrtOp,
        SinOp,
        SinhOp,
        SqrtOp,
        TanOp,
        TanhOp,
        TruncOp,
        Atan2Op,
        CopySignOp,
        PowFOp,
        FmaOp,
        ClampFOp,
        AbsIOp,
        CountLeadingZerosOp,
        CountTrailingZerosOp,
        CtPopOp,
        IPowIOp,
        IsFiniteOp,
        IsInfOp,
        IsNaNOp,
        IsNormalOp,
        SincosOp,
    ),
  ]

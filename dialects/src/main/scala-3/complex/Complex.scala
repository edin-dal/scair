package scair.dialects.complex

import fastparse.*
import scair.AttrParser
import scair.Printer
import scair.clair.codegen.*
import scair.clair.macros.*
import scair.dialects.arith.*
import scair.dialects.builtin.*
import scair.ir.*

case class Abs(
    complex: Operand[ComplexType],
    result: Result[FloatType]
) extends DerivedOperation["complex.abs", Abs]
    with NoMemoryEffect

case class Add(
    lhs: Operand[ComplexType],
    rhs: Operand[ComplexType],
    result: Result[ComplexType],
    fastmath: FastMathFlagsAttr
) extends DerivedOperation["complex.add", Add]
    with NoMemoryEffect
    // TODO: Should probably handle fastmath flags!
    with Commutative

case class Constant(
    value: ArrayAttribute[IntegerAttr | FloatAttr],
    complex: Result[ComplexType]
) extends DerivedOperation["complex.constant", Constant]
    with NoMemoryEffect
    with ConstantLike

case class Create(
    real: Operand[IndexType | IntegerType | FloatType],
    imaginary: Operand[IndexType | IntegerType | FloatType],
    complex: Result[ComplexType]
) extends DerivedOperation["complex.create", Create]
    with NoMemoryEffect

case class Div(
    lhs: Operand[ComplexType],
    rhs: Operand[ComplexType],
    result: Result[ComplexType],
    fastmath: FastMathFlagsAttr
) extends DerivedOperation["complex.div", Div]
    with NoMemoryEffect

case class Im(
    complex: Operand[ComplexType],
    imaginary: Result[IndexType | IntegerType | FloatType]
) extends DerivedOperation["complex.im", Im]
    with NoMemoryEffect

case class Mul(
    lhs: Operand[ComplexType],
    rhs: Operand[ComplexType],
    result: Result[ComplexType],
    fastmath: FastMathFlagsAttr
) extends DerivedOperation["complex.mul", Mul]
    with NoMemoryEffect
    with Commutative

case class Neg(
    complex: Operand[ComplexType],
    result: Result[ComplexType],
    fastmath: FastMathFlagsAttr
) extends DerivedOperation["complex.neg", Neg]
    with NoMemoryEffect

case class Re(
    complex: Operand[ComplexType],
    real: Result[IndexType | IntegerType | FloatType]
) extends DerivedOperation["complex.re", Re]
    with NoMemoryEffect

case class Sub(
    lhs: Operand[ComplexType],
    rhs: Operand[ComplexType],
    result: Result[ComplexType],
    fastmath: FastMathFlagsAttr
) extends DerivedOperation["complex.sub", Sub]
    with NoMemoryEffect

object ComplexAttr:

  def parse[$: P](parser: AttrParser): P[ComplexAttr] =
    given Whitespace = scair.Parser.whitespace
    ("<:" ~ parser.FloatTypeP ~ parser.FloatDataP ~ "," ~ parser.FloatDataP ~ ">" ~ (":" ~ parser.ComplexTypeP).?)
      .map(
        (
            tpe: FloatType,
            real: FloatData,
            imaginary: FloatData,
            ctpe: Option[ComplexType]
        ) => ComplexAttr(real, imaginary, ctpe.getOrElse(ComplexType(tpe)))
      )

case class ComplexAttr(
    real: FloatData,
    imaginary: FloatData,
    tpe: ComplexType
) extends DerivedAttribute["complex.number", ComplexAttr]:

  override def custom_print(p: Printer): Unit =
    p.print(
      s"#complex.number<:",
      tpe.tpe,
      " ",
      real,
      ", ",
      imaginary,
      "> : ",
      tpe
    )(using 0)

val Complex = summonDialect[Tuple1[
  ComplexAttr
], (Abs, Add, Constant, Create, Div, Im, Mul, Neg, Re, Sub)](Seq())

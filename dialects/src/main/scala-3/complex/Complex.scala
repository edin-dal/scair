package scair.dialects.complex

import fastparse.*
import scair.AttrParser
import scair.Printer
import scair.clair.codegen.*
import scair.clair.macros.*
import scair.dialects.arith.*
import scair.dialects.builtin.*
import scair.dialects.complex.canonicalization.given
import scair.ir.*

//
// ░█████╗░ ░█████╗░ ███╗░░░███╗ ██████╗░ ██╗░░░░░ ███████╗ ██╗░░██╗
// ██╔══██╗ ██╔══██╗ ████╗░████║ ██╔══██╗ ██║░░░░░ ██╔════╝ ╚██╗██╔╝
// ██║░░╚═╝ ██║░░██║ ██╔████╔██║ ██████╔╝ ██║░░░░░ █████╗░░ ░╚███╔╝░
// ██║░░██╗ ██║░░██║ ██║╚██╔╝██║ ██╔═══╝░ ██║░░░░░ ██╔══╝░░ ░██╔██╗░
// ╚█████╔╝ ╚█████╔╝ ██║░╚═╝░██║ ██║░░░░░ ███████╗ ███████╗ ██╔╝╚██╗
// ░╚════╝░ ░╚════╝░ ╚═╝░░░░░╚═╝ ╚═╝░░░░░ ╚══════╝ ╚══════╝ ╚═╝░░╚═╝
//

case class Abs(
    complex: Operand[ComplexType],
    result: Result[FloatType]
) extends DerivedOperation["complex.abs", Abs]
    with NoMemoryEffect derives DerivedOperationCompanion

case class Add(
    lhs: Operand[ComplexType],
    rhs: Operand[ComplexType],
    result: Result[ComplexType],
    fastmath: FastMathFlagsAttr
) extends DerivedOperation["complex.add", Add]
    with NoMemoryEffect
    // TODO: Should probably handle fastmath flags!
    with Commutative derives DerivedOperationCompanion

case class Constant(
    value: ArrayAttribute[IntegerAttr | FloatAttr],
    complex: Result[ComplexType]
) extends DerivedOperation["complex.constant", Constant]
    with AssemblyFormat["$value attr-dict `:` type($complex)"]
    with NoMemoryEffect
    with ConstantLike(value) derives DerivedOperationCompanion:

  // The complex dialect is really acting weird with that.
  // 1. The attribute in the constant op is not the general "complex value" attribute
  // 2. The general "complex value" attribute is restricted to float, the constant is not.
  // (Is that why the mismatch? Either way, here's a hacky workaround for now.)
  override def getValue: ComplexAttr =
    val real = value(0).asInstanceOf[FloatAttr].value
    val imaginary = value(1).asInstanceOf[FloatAttr].value
    ComplexAttr(real, imaginary, complex.typ)

case class Create(
    real: Operand[IndexType | IntegerType | FloatType],
    imaginary: Operand[IndexType | IntegerType | FloatType],
    complex: Result[ComplexType]
) extends DerivedOperation["complex.create", Create]
    with NoMemoryEffect derives DerivedOperationCompanion

case class Div(
    lhs: Operand[ComplexType],
    rhs: Operand[ComplexType],
    result: Result[ComplexType],
    fastmath: FastMathFlagsAttr
) extends DerivedOperation["complex.div", Div]
    with NoMemoryEffect derives DerivedOperationCompanion

case class Im(
    complex: Operand[ComplexType],
    imaginary: Result[IndexType | IntegerType | FloatType]
) extends DerivedOperation["complex.im", Im]
    with NoMemoryEffect derives DerivedOperationCompanion

case class Mul(
    lhs: Operand[ComplexType],
    rhs: Operand[ComplexType],
    result: Result[ComplexType],
    fastmath: FastMathFlagsAttr
) extends DerivedOperation["complex.mul", Mul]
    with NoMemoryEffect
    with Commutative derives DerivedOperationCompanion

case class Neg(
    complex: Operand[ComplexType],
    result: Result[ComplexType],
    fastmath: FastMathFlagsAttr
) extends DerivedOperation["complex.neg", Neg]
    with NoMemoryEffect derives DerivedOperationCompanion

case class Re(
    complex: Operand[ComplexType],
    real: Result[IndexType | IntegerType | FloatType]
) extends DerivedOperation["complex.re", Re]
    with NoMemoryEffect derives DerivedOperationCompanion

case class Sub(
    lhs: Operand[ComplexType],
    rhs: Operand[ComplexType],
    result: Result[ComplexType],
    fastmath: FastMathFlagsAttr
) extends DerivedOperation["complex.sub", Sub]
    with NoMemoryEffect derives DerivedOperationCompanion

object ComplexAttr:

  def parse[$: P](parser: AttrParser): P[ComplexAttr] =
    given Whitespace = scair.AttrParser.whitespace
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
) extends DerivedAttribute["complex.number", ComplexAttr]
    derives DerivedAttributeCompanion:

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
], (Abs, Add, Constant, Create, Div, Im, Mul, Neg, Re, Sub)]

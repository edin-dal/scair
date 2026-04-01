package scair.dialects.complex

import fastparse.*
import scair.Printer
import scair.clair.*
import scair.dialects.arith.*
import scair.dialects.builtin.*
import scair.dialects.complex.canonicalization.given
import scair.ir.*
import scair.parse.*

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
    result: Result[FloatType],
) extends DerivedOperation["complex.abs"]
    with NoMemoryEffect derives OpDefs

case class Add(
    lhs: Operand[ComplexType],
    rhs: Operand[ComplexType],
    result: Result[ComplexType],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
) extends DerivedOperation["complex.add"]
    with NoMemoryEffect
    // TODO: Should probably handle fastmath flags!
    with Commutative derives OpDefs

case class Constant(
    value: ArrayAttribute[IntegerAttr | FloatAttr],
    complex: Result[ComplexType],
) extends DerivedOperation["complex.constant"]
    with AssemblyFormat["$value attr-dict `:` type($complex)"]
    with NoMemoryEffect
    with ConstantLike(value) derives OpDefs:

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
    complex: Result[ComplexType],
) extends DerivedOperation["complex.create"]
    with NoMemoryEffect derives OpDefs

case class Div(
    lhs: Operand[ComplexType],
    rhs: Operand[ComplexType],
    result: Result[ComplexType],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
) extends DerivedOperation["complex.div"]
    with NoMemoryEffect derives OpDefs

case class Im(
    complex: Operand[ComplexType],
    imaginary: Result[IndexType | IntegerType | FloatType],
) extends DerivedOperation["complex.im"]
    with NoMemoryEffect derives OpDefs

case class Mul(
    lhs: Operand[ComplexType],
    rhs: Operand[ComplexType],
    result: Result[ComplexType],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
) extends DerivedOperation["complex.mul"]
    with NoMemoryEffect
    with Commutative derives OpDefs

case class Neg(
    complex: Operand[ComplexType],
    result: Result[ComplexType],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
) extends DerivedOperation["complex.neg"]
    with NoMemoryEffect derives OpDefs

case class Re(
    complex: Operand[ComplexType],
    real: Result[IndexType | IntegerType | FloatType],
) extends DerivedOperation["complex.re"]
    with NoMemoryEffect derives OpDefs

case class Sub(
    lhs: Operand[ComplexType],
    rhs: Operand[ComplexType],
    result: Result[ComplexType],
    fastmath: FastMathFlagsAttr = FastMathFlagsAttr(FastMathFlags.none),
) extends DerivedOperation["complex.sub"]
    with NoMemoryEffect derives OpDefs

given AttributeCustomParser[ComplexAttr]:

  def parse[$: P](using Parser): P[ComplexAttr] =
    given Whitespace = scair.parse.whitespace
    ("<:" ~ floatTypeP ~ floatDataP ~ "," ~ floatDataP ~ ">" ~
      (":" ~ complexTypeP).?)
      .map(
        (
            tpe: FloatType,
            real: FloatData,
            imaginary: FloatData,
            ctpe: Option[ComplexType],
        ) => ComplexAttr(real, imaginary, ctpe.getOrElse(ComplexType(tpe)))
      )

case class ComplexAttr(
    real: FloatData,
    imaginary: FloatData,
    tpe: ComplexType,
) extends DerivedAttribute["complex.number"] derives AttrDefs:

  override def customPrint(p: Printer): Unit =
    p.print(
      s"#complex.number<:",
      tpe.tpe,
      " ",
      real,
      ", ",
      imaginary,
      "> : ",
      tpe,
    )

val Complex = summonDialect[Tuple1[
  ComplexAttr
], (Abs, Add, Constant, Create, Div, Im, Mul, Neg, Re, Sub)]

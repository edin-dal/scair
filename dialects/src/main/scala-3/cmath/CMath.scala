package scair.dialects.cmath

import scair.clair.codegen.*
import scair.clair.mirrored.*
import scair.clair.macros._
import scair.dialects.builtin.*
import scair.ir.*

object Complex extends AttributeObject {
  override def name: String = "cmath.complex"
  override def factory = Complex.apply
}

case class Complex(
    val typ: Seq[Attribute]
) extends ParametrizedAttribute(
      name = "cmath.complex",
      parameters = Seq(typ)
    )
    with TypeAttribute

case class Norm(
    in: Operand[Complex],
    res: Result[FloatType]
) extends MLIRName["cmath.norm"]
    derives MLIRTrait

case class Mul(
    lhs: Operand[Complex],
    rhs: Operand[Complex],
    res: Result[Complex]
) extends MLIRName["cmath.mul"]
    derives MLIRTrait

val CMathDialect = summonDialect[(Norm, Mul)](Seq(Complex))

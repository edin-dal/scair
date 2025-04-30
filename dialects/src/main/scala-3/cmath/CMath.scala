package scair.dialects.cmath

import scair.clair.codegen.*
import scair.clair.macros.*
import scair.core.constraint.*
import scair.core.constraint.given_ConstrainedCompanion_A_EqualAttr
import scair.dialects.builtin.*
import scair.ir.*

case class Complex(
    val typ: FloatType | IndexType.type
) extends ParametrizedAttribute(
      name = "cmath.complex",
      parameters = Seq(typ)
    )
    with TypeAttribute
    with MLIRName["cmath.complex"]
    derives AttributeTrait

val cf16 = Complex(Float16Type)

case class Norm(
    iinn: Operand[Complex Constrained EqualAttr[cf16.type]],
    res: Result[FloatType]
) extends DerivedOperation["cmath.norm", Norm]
    derives DerivedOperationCompanion

case class Mul(
    lhs: Operand[Complex],
    rhs: Operand[Complex],
    res: Result[Complex]
) extends DerivedOperation["cmath.mul", Mul]
    derives DerivedOperationCompanion

val CMathDialect = summonDialect[Tuple1[Complex], (Norm, Mul)](Seq())

object DaMain {

  def main(args: Array[String]): Unit = {
    val n = UnverifiedOp[Norm](
      "cmath.norm",
      Seq(new Value(Complex(Float32Type))),
      results_types = Seq(Complex(Float16Type))
    )
    summon[DerivedOperationCompanion[Norm]].verify(n)
  }

}

package scair.dialects.cmath

import scair.clair.codegen.*
import scair.clair.macros.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.clair.constraints.*
case class Complex(
    val typ: FloatType | IndexType
) extends DerivedAttribute["cmath.complex", Complex]
    with TypeAttribute derives DerivedAttributeCompanion

case class Norm(
    in: Operand[Complex],
    res: Result[FloatType]
) extends DerivedOperation["cmath.norm", Norm] derives DerivedOperationCompanion

case class Mul(
    lhs: Operand[Complex],
    rhs: Operand[Complex],
    res: Result[Complex]
) extends DerivedOperation["cmath.mul", Mul] derives DerivedOperationCompanion

val CMathDialect = summonDialect[Tuple1[Complex], (Norm, Mul)](Seq())

object ConstraintKitchen {
    def main(args: Array[String]): Unit = {

        val a: IntegerType = IntegerType(IntData(32), Signless)
        val b: IntegerType = IntegerType(IntData(16), Signless)
        val c: IntegerType = IntegerType(IntData(32), Signless)

        // Example usage
        Eq.verify[IntegerType, a.type](b)

        // println(eq1)
    }
}
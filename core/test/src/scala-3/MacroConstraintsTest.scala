package scair

import org.scalatest.*
import org.scalatest.flatspec.*
import scair.clair.codegen.*
import scair.clair.macros.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.core.constraints.{_, given}

val f32 = Float32Type()

case class MulF(
    lhs: Operand[FloatType !> EqAttr[f32.type]],
    rhs: Operand[FloatType !> EqAttr[f32.type]],
    result: Result[FloatType]
) extends DerivedOperation["cmath.mul", MulF]

class MacroConstraintsTest extends AnyFlatSpec with BeforeAndAfter {
  val a = Value[FloatType](Float64Type())
  val b = Value[FloatType](Float64Type())
  val c = Value[FloatType](Float64Type())

  val x = MulF(a, b, c)

  "A derived operation with constraints" should "fail verification if constraints are not met" in {
    val res = x.verify()
    assert(res.isLeft)
    assert(
      res.left.get.contains(
        s"Expected ${f32}, got ${Float64Type()}"
      )
    )
  }

}

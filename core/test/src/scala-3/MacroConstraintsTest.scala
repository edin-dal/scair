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

  "A derived operation with constraints" should "fail verification if constraints are not met" in {
    val x = MulF(
      Value[FloatType](Float64Type()),
      Value[FloatType](Float64Type()),
      Value[FloatType](Float64Type())
    )

    val res = x.verify()
    assert(res.isLeft)
    assert(
      res.left.get.contains(
        s"Expected ${f32}, got ${Float64Type()}"
      )
    )
  }

  "A derived operation with constraints" should "verify if constraints are met" in {
    val x = MulF(
      Value[FloatType](Float32Type()),
      Value[FloatType](Float32Type()),
      Value[FloatType](Float32Type())
    )

    val res = x.verify()
    assert(res.isRight)
  }

}

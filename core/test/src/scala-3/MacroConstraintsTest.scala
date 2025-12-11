package scair

import org.scalatest.*
import org.scalatest.flatspec.*
import scair.clair.codegen.*
import scair.clair.macros.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.core.constraints.{*, given}

val f32 = Float32Type()

type T = Var["T"]

case class MulFEq(
    lhs: Operand[FloatType !> EqAttr[f32.type]],
    rhs: Operand[FloatType !> EqAttr[f32.type]],
    result: Result[FloatType],
) extends DerivedOperation["cmath.mul", MulFEq]
    derives DerivedOperationCompanion

case class MulFVar(
    lhs: Operand[FloatType !> T],
    rhs: Operand[FloatType !> T],
    result: Result[FloatType],
) extends DerivedOperation["cmath.mul", MulFVar]
    derives DerivedOperationCompanion

class MacroConstraintsTest extends AnyFlatSpec with BeforeAndAfter:

  "A derived operation with eq-attr constraints" should
    "fail verification if constraints are not met" in {
      val x = MulFEq(
        Value[FloatType](Float64Type()),
        Value[FloatType](Float64Type()),
        Value[FloatType](Float64Type()),
      )

      val res = x.verify()
      assert(res.isLeft)
      assert(
        res.left.get
          .contains(
            s"Expected $f32, got ${Float64Type()}"
          )
      )
    }

  "A derived operation with eq-attr constraints" should
    "verify if constraints are met" in {
      val x = MulFEq(
        Value[FloatType](Float32Type()),
        Value[FloatType](Float32Type()),
        Value[FloatType](Float32Type()),
      )

      val res = x.verify()
      assert(res.isRight)
    }

  "A derived operation with var-def constraints" should
    "fail verification if constraints are not met - 1" in {
      val x = MulFVar(
        Value[FloatType](Float32Type()),
        Value[FloatType](Float64Type()),
        Value[FloatType](Float64Type()),
      )

      val res = x.verify()
      assert(res.isLeft)
      assert(
        res.left.get
          .contains(
            s"Expected ${Float32Type()}, got ${Float64Type()}"
          )
      )
    }

  it should "fail verification if constraints are not met - 2" in {
    val x = MulFVar(
      Value[FloatType](Float64Type()),
      Value[FloatType](Float32Type()),
      Value[FloatType](Float64Type()),
    )

    val res = x.verify()
    assert(res.isLeft)
    assert(
      res.left.get
        .contains(
          s"Expected ${Float64Type()}, got ${Float32Type()}"
        )
    )
  }

  it should "fail verification if constraints are not met - 3" in {
    val x = MulFVar(
      Value[FloatType](Float32Type()),
      Value[FloatType](Float64Type()),
      Value[FloatType](Float64Type()),
    )

    val res = x.verify()
    assert(res.isLeft)
    assert(
      res.left.get
        .contains(
          s"Expected ${Float32Type()}, got ${Float64Type()}"
        )
    )
  }

  "A derived operation with var-def constraints" should
    "verify if constraints are met" in {
      val x = MulFVar(
        Value[FloatType](Float32Type()),
        Value[FloatType](Float32Type()),
        Value[FloatType](Float32Type()),
      )

      val res = x.verify()
      assert(res.isRight)
    }

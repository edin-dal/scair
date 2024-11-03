package scair

import org.scalatest._
import flatspec._
import matchers.should.Matchers._
import prop._

import fastparse._, MultiLineWhitespace._
import scala.collection.mutable
import scala.util.{Try, Success, Failure}
import org.scalatest.prop.Tables.Table
import org.scalatest.prop.TableDrivenPropertyChecks.forAll

import AttrParser._
import Parser._
import scair.dialects.builtin._
import scair.ir._
import scair.scairdl.constraints._
import scair.scairdl.constraints.attr2constraint
class ConstraintsTest extends AnyFlatSpec with BeforeAndAfter {

  var constraint_ctx = new ConstraintContext()

  before {
    constraint_ctx = new ConstraintContext()
  }

  ///////////
  //  ANY  //
  ///////////

  "Any Constraint Test" should "test AnyAttr constraint" in {
    val any_constraint = AnyAttr
    any_constraint.verify(Float16Type, constraint_ctx)
    any_constraint.verify(Float32Type, constraint_ctx)
  }

  /////////////
  //  EQUAL  //
  /////////////

  "Equal Constraint Test 1" should "verify EqualAttr constraint (should PASS)" in {

    val this_attr = ArrayAttribute(Seq(IndexType, Float16Type, I64))
    val that_attr = ArrayAttribute(Seq(IndexType, Float16Type, I64))

    val equal_constraint = EqualAttr(this_attr)

    equal_constraint.verify(that_attr, constraint_ctx)
  }

  "Equal Constraint Test 2" should "not verify EqualAttr constraint (should FAIL)" in {

    val this_attr = ArrayAttribute(Seq(IndexType, Float16Type, I64))
    val that_attr = ArrayAttribute(Seq(IndexType, Float32Type, I64))

    val equal_constraint = EqualAttr(this_attr)

    val exception = intercept[Exception](
      equal_constraint.verify(that_attr, constraint_ctx)
    ).getMessage shouldBe
      "builtin.array_attr does not equal builtin.array_attr:\n" +
      "Got [index, f32, i64], expected [index, f16, i64]"
  }

  "Equal Constraint Test 3" should "verify EqualAttr constraint (should PASS)" in {

    val attr1 = IntegerAttr(IntData(5), I32)
    val attr2 = IntegerAttr(IntData(5), I32)

    val this_attr = ArrayAttribute(Seq(IndexType, attr1, I64))
    val that_attr = ArrayAttribute(Seq(IndexType, attr2, I64))

    val equal_constraint = EqualAttr(this_attr)

    equal_constraint.verify(that_attr, constraint_ctx)
  }

  "Equal Constraint Test 4" should "not verify EqualAttr constraint (should FAIL)" in {

    val attr1 = IntegerAttr(IntData(5), I32)
    val attr2 = IntegerAttr(IntData(6), I32)

    val this_attr = ArrayAttribute(Seq(IndexType, attr1, I64))
    val that_attr = ArrayAttribute(Seq(IndexType, attr2, I64))

    val equal_constraint = EqualAttr(this_attr)

    val exception = intercept[Exception](
      equal_constraint.verify(that_attr, constraint_ctx)
    ).getMessage shouldBe
      "builtin.array_attr does not equal builtin.array_attr:\n" +
      "Got [index, 6 : i32, i64], expected [index, 5 : i32, i64]"
  }

  ////////////
  //  BASE  //
  ////////////

  "Base Constraint Test 1" should "verify BaseAttr constraint (should PASS)" in {

    val that_attr = ArrayAttribute(Seq(IndexType, Float32Type, I64))

    val base_constraint = BaseAttr[ArrayAttribute[Attribute]]()

    base_constraint.verify(that_attr, constraint_ctx)
  }

  "Base Constraint Test 2" should "not verify BaseAttr constraint (should FAIL)" in {

    val that_attr = IndexType

    val base_constraint = BaseAttr[ArrayAttribute[IntData]]()

    val exception = intercept[Exception](
      base_constraint.verify(that_attr, constraint_ctx)
    ).getMessage shouldBe
      "builtin.index's class does not equal scair.dialects.builtin.ArrayAttribute\n"
  }

  /////////////
  //  ANYOF  //
  /////////////

  "OR Constraint Test 1" should "verify AnyOf constraint (should PASS)" in {

    val that_attr = ArrayAttribute(Seq(IndexType, IndexType, I64))

    val attr1 = IntegerAttr(IntData(5), I32)
    val attr2 = IntegerAttr(IntData(6), I32)
    val attr3 = Float32Type
    val attr4 = IndexType
    val attr5 = ArrayAttribute(Seq(IndexType, attr4, I64))

    val or_constraint = AnyOf(Seq(attr1, attr2, attr3, attr4, attr5))

    or_constraint.verify(that_attr, constraint_ctx)
  }

  "OR Constraint Test 2" should "not verify AnyOf constraint (should FAIL)" in {

    val that_attr =
      ArrayAttribute(Seq(IndexType, IntegerAttr(IntData(5), I32), I64))

    val attr1 = IntegerAttr(IntData(5), I32)
    val attr2 = IntegerAttr(IntData(6), I32)
    val attr3 = Float32Type
    val attr4 = IndexType

    val or_constraint = AnyOf(Seq(attr1, attr2, attr3, attr4))

    val exception = intercept[Exception](
      or_constraint.verify(that_attr, constraint_ctx)
    ).getMessage shouldBe
      "builtin.array_attr does not match any of List(EqualAttr(5 : i32), EqualAttr(6 : i32), EqualAttr(f32), EqualAttr(index))\n"
  }

  "OR Constraint Test 3" should "verify AnyOf constraint (should PASS)" in {

    val that_attr = ArrayAttribute(Seq(IndexType, IndexType, I64))

    val attr1 = IntegerAttr(IntData(5), I32)
    val attr2 = IntegerAttr(IntData(6), I32)
    val attr3 = Float32Type
    val attr4 = IndexType
    val attrcnst1 = BaseAttr[IndexType.type]()
    val attrcnst2 = EqualAttr(ArrayAttribute(Seq(IndexType, IndexType, I64)))

    val or_constraint = AnyOf(Seq(attr1, attr2, attr3, attrcnst1, attrcnst2))

    or_constraint.verify(that_attr, constraint_ctx)
  }

  "OR Constraint Test 4" should "not verify AnyOf constraint (should FAIL)" in {

    val that_attr =
      ArrayAttribute(Seq(IndexType, IntegerAttr(IntData(5), I32), I64))

    val attr1 = IntegerAttr(IntData(5), I32)
    val attr2 = IntegerAttr(IntData(6), I32)
    val attr3 = Float32Type
    val attr4 = IndexType
    val attrcnst1 = BaseAttr[IndexType.type]()
    val attrcnst2 = EqualAttr(ArrayAttribute(Seq(IndexType, I32, I64)))

    val or_constraint =
      AnyOf(Seq(attr1, attr2, attr3, attr4, attrcnst1, attrcnst2))

    val exception = intercept[Exception](
      or_constraint.verify(that_attr, constraint_ctx)
    ).getMessage should include
    "builtin.array_attr does not match any of List(EqualAttr(5 : i32), EqualAttr(6 : i32), EqualAttr(f32), EqualAttr(index), BaseAttr(), EqualAttr([index, i32, i64]))"
  }

  //////////////////////
  //  ParametrizedAttrConstraint  //
  //////////////////////

  "Parametric Constraint Test 1" should "verify ParametrizedAttrConstraint constraint (should PASS)" in {

    val that_attr = FloatAttr(FloatData(1.5), Float32Type)

    val parametric_constraint =
      ParametrizedAttrConstraint[FloatAttr](Seq(FloatData(1.5), Float32Type))

    parametric_constraint.verify(that_attr, constraint_ctx)
  }

  "Parametric Constraint Test 2" should "not verify ParametrizedAttrConstraint constraint (should FAIL)" in {

    val that_attr = ArrayAttribute(Seq(IndexType, Float32Type, I64))

    val parametric_constraint =
      ParametrizedAttrConstraint[ArrayAttribute[Attribute]](
        Seq(IndexType, I32, I64)
      )

    val exception = intercept[Exception](
      parametric_constraint.verify(that_attr, constraint_ctx)
    ).getMessage shouldBe
      "Attribute being verified must be of type ParametrizedAttribute.\n"
  }

  "Parametric Constraint Test 3" should "not verify ParametrizedAttrConstraint constraint (should FAIL)" in {

    val that_attr = FloatAttr(FloatData(1.5), Float32Type)

    val parametric_constraint =
      ParametrizedAttrConstraint[FloatAttr](Seq(FloatData(1.6), Float32Type))

    val exception = intercept[Exception](
      parametric_constraint.verify(that_attr, constraint_ctx)
    ).getMessage shouldBe
      "builtin.float_data does not equal builtin.float_data:\nGot 1.5, expected 1.6"
  }

  "Parametric Constraint Test 4" should "not verify ParametrizedAttrConstraint constraint (should FAIL)" in {

    val that_attr = FloatAttr(FloatData(1.5), Float32Type)

    val parametric_constraint =
      ParametrizedAttrConstraint[IntegerAttr](Seq(FloatData(1.5), Float32Type))

    val exception = intercept[Exception](
      parametric_constraint.verify(that_attr, constraint_ctx)
    ).getMessage shouldBe
      "builtin.float_attr's class does not equal scair.dialects.builtin.IntegerAttr.\n"
  }

  /////////////////////
  //  VARCONSTRAINT  //
  /////////////////////

  "Variable Constraint Test 1" should "verify VarConstraint constraint (should PASS)" in {

    val that_attr = FloatAttr(FloatData(1.5), Float32Type)

    val parametric_constraint =
      ParametrizedAttrConstraint[FloatAttr](Seq(FloatData(1.5), Float32Type))

    val var_constraint = VarConstraint("T", parametric_constraint)

    var_constraint.verify(that_attr, constraint_ctx)

    constraint_ctx.var_constraints.apply("T") shouldBe that_attr
  }

  "Variable Constraint Test 2" should "not verify VarConstraint constraint (should FAIL)" in {

    val that_attr = FloatAttr(FloatData(1.5), Float32Type)
    val that_attr2 = FloatAttr(FloatData(1.6), Float32Type)

    val parametric_constraint =
      ParametrizedAttrConstraint[FloatAttr](Seq(FloatData(1.5), Float32Type))

    val var_constraint = VarConstraint("T", parametric_constraint)

    var_constraint.verify(that_attr, constraint_ctx)

    val exception = intercept[Exception](
      var_constraint.verify(that_attr2, constraint_ctx)
    ).getMessage shouldBe
      "oh mah gawd"
  }

  "Variable Constraint Test 3" should "not verify VarConstraint constraint (should FAIL)" in {

    val that_attr = FloatAttr(FloatData(1.6), Float32Type)

    val parametric_constraint =
      ParametrizedAttrConstraint[FloatAttr](Seq(FloatData(1.5), Float32Type))

    val var_constraint = VarConstraint("T", parametric_constraint)

    val exception = intercept[Exception](
      var_constraint.verify(that_attr, constraint_ctx)
    ).getMessage shouldBe
      "builtin.float_data does not equal builtin.float_data:\nGot 1.6, expected 1.5"
  }
}

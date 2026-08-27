package scair.constrainttest

import org.scalatest.flatspec.*
import scair.*
import scair.clair.*
import scair.clair.macros.*
import scair.constraints.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.utils.*

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||   ATTRIBUTES USED BY THE FIXTURES   ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

val f16 = Float16Type()
val f32 = Float32Type()
val f64 = Float64Type()
val w32 = IntData(32)
val i32 = IntegerType(w32, Signless)
val i64 = IntegerType(IntData(64), Signless)

type Float32 = EqAttr[f32.type]
type Float64 = EqAttr[f64.type]
type AnyFloat = Base[Float16Type] || Float32 || Float64

/*≡==--==≡≡≡≡≡≡≡≡==--=≡≡*\
||      FIXTURE OPS     ||
\*≡==---==≡≡≡≡≡≡==---==≡*/

case class BaseOp(
    x: Operand[Attribute !> Base[FloatType]]
) extends DerivedOperation["t.base"] derives OpDefs

case class AnyOfOp(
    x: Operand[Attribute !> (Float32 || Float64)]
) extends DerivedOperation["t.anyof"] derives OpDefs

case class AllOfOp(
    x: Operand[Attribute !> (Base[FloatType] && Float32)]
) extends DerivedOperation["t.allof"] derives OpDefs

case class ParamOp(
    x: Operand[Attribute !> Param[IntegerType, (EqAttr[w32.type], AnyAttr)]]
) extends DerivedOperation["t.param"] derives OpDefs

case class ParamVarOp(
    x: Operand[Attribute !> Param[IntegerType, (Var["W"], AnyAttr)]],
    y: Operand[Attribute !> Param[IntegerType, (Var["W"], AnyAttr)]],
) extends DerivedOperation["t.paramvar"] derives OpDefs

case class MsgOp(
    x: Operand[Attribute !> Msg["x must be single precision", Float32]]
) extends DerivedOperation["t.msg"] derives OpDefs

case class SameTypeOp(
    x: Operand[Attribute !> Var["T"]],
    res: Result[Attribute !> Var["T"]],
) extends DerivedOperation["t.sametype"] derives OpDefs

/** `T` is first bound inside an AnyOf alternative, so it cannot be a
  * compile-time expression and gets a runtime slot.
  */
case class AnyOfVarOp(
    x: Operand[
      Attribute !>
        ((Base[FloatType] && Var["T"]) || (Base[IntegerType] && Var["T"]))
    ],
    y: Operand[Attribute !> Var["T"]],
) extends DerivedOperation["t.anyofvar"] derives OpDefs

/** The first alternative binds `T` and *then* fails, so the binding has to be
  * rolled back before the second alternative is tried.
  */
case class AnyOfRollbackOp(
    x: Operand[
      Attribute !> ((Var["T"] && Base[FloatType]) || Base[IntegerType])
    ],
    y: Operand[Attribute !> Var["T"]],
) extends DerivedOperation["t.anyofrollback"] derives OpDefs

case class NestedOp(
    x: Operand[Attribute !> AnyFloat]
) extends DerivedOperation["t.nested"] derives OpDefs

/*≡==--==≡≡≡≡≡≡≡==--=≡≡*\
||        TESTS        ||
\*≡==---==≡≡≡≡≡==---==≡*/

class ConstraintsTest extends AnyFlatSpec:

  def v(a: Attribute) = Value[Attribute](a)

  "Base" should "accept a subtype" in assert(BaseOp(v(f32)).verify().isOK)

  it should "reject an unrelated attribute" in {
    val r = BaseOp(v(i32)).verify()
    assert(r.isError)
    assert(r.getError.msg.contains("operand 'x'"))
  }

  "AnyOf" should "accept the first alternative" in
    assert(AnyOfOp(v(f32)).verify().isOK)

  it should "accept a later alternative" in
    assert(AnyOfOp(v(f64)).verify().isOK)

  it should "reject anything else, naming the alternatives" in {
    val r = AnyOfOp(v(i32)).verify()
    assert(r.isError)
    assert(r.getError.msg.contains("one of"))
  }

  it should "nest" in {
    assert(NestedOp(v(f16)).verify().isOK)
    assert(NestedOp(v(f64)).verify().isOK)
    assert(NestedOp(v(i32)).verify().isError)
  }

  it should "bind a variable from the alternative that ran" in {
    assert(AnyOfVarOp(v(f32), v(f32)).verify().isOK)
    assert(AnyOfVarOp(v(i32), v(i32)).verify().isOK)
  }

  it should "propagate that binding to later constructs" in {
    assert(AnyOfVarOp(v(f32), v(f64)).verify().isError)
    assert(AnyOfVarOp(v(i32), v(f32)).verify().isError)
  }

  it should "roll back a binding made by a failing alternative" in
    // The first alternative binds T := i32 and only then fails on
    // Base[FloatType]. If that binding survived, y would be checked against
    // i32 and this would fail.
    assert(AnyOfRollbackOp(v(i32), v(f32)).verify().isOK)

  "AllOf" should "accept when every conjunct holds" in
    assert(AllOfOp(v(f32)).verify().isOK)

  it should "short-circuit on the first failure" in {
    // i32 fails Base[FloatType]; the EqAttr message must not appear.
    val r = AllOfOp(v(i32)).verify()
    assert(r.isError)
    assert(r.getError.msg.contains("FloatType"))
    assert(!r.getError.msg.contains("f32"))
  }

  it should "report a later conjunct when earlier ones hold" in {
    val r = AllOfOp(v(f64)).verify()
    assert(r.isError)
    assert(r.getError.msg.contains("Expected f32, got f64"))
  }

  "Param" should "check an attribute's parameters" in
    assert(ParamOp(v(i32)).verify().isOK)

  it should "reject a bad parameter, naming it" in {
    val r = ParamOp(v(i64)).verify()
    assert(r.isError)
    assert(r.getError.msg.contains("operand 'x' parameter 'width'"))
  }

  it should "reject a wrong base" in {
    val r = ParamOp(v(f32)).verify()
    assert(r.isError)
    assert(r.getError.msg.contains("IntegerType"))
  }

  it should "tie a Var nested inside a parameter" in {
    assert(ParamVarOp(v(i32), v(i32)).verify().isOK)
    assert(ParamVarOp(v(i32), v(i64)).verify().isError)
  }

  "Msg" should "replace the reported expectation" in {
    val r = MsgOp(v(f64)).verify()
    assert(r.isError)
    assert(r.getError.msg.contains("x must be single precision"))
    assert(r.getError.msg.contains("Underlying verification failure"))
  }

  "A Var" should "tie an operand to a result" in
    assert(SameTypeOp(v(f32), Result(f32)).verify().isOK)

  it should "reject a result that disagrees with the operand" in {
    val r = SameTypeOp(v(f32), Result(f64)).verify()
    assert(r.isError)
    assert(r.getError.msg.contains("result 'res'"))
    assert(r.getError.msg.contains("Expected f32, got f64"))
  }

package scair.dialects

import org.scalatest.flatspec.*
import scair.clair.*
import scair.clair.macros.*
import scair.constraints.*
import scair.dialects.builtin.*
import scair.dialects.constraints.*
import scair.ir.*
import scair.parse.*
import scair.utils.*
import scair.MLContext

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||   USING AN OUT-OF-TREE CONSTRAINT GEN   ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/
//
// `Width` and its generator live in the `dialects` module, this file is
// compiled against it: the generator really does run during *this* file's macro
// expansion, from a separate compilation unit.

case class WidthOp(
    x: Operand[Attribute !> Width[32]]
) extends DerivedOperation["ext.width"] derives OpDefs

case class WidthInferOp(
    x: Operand[Attribute !> Width[64]],
    res: Result[Attribute !> Width[64]],
) extends DerivedOperation["ext.width_infer"]
    with AssemblyFormat["$x attr-dict `:` type($x)"] derives OpDefs

val ExtDialect = summonDialect[EmptyTuple, (WidthOp, WidthInferOp)]

class ExampleConstraintsTest extends AnyFlatSpec:

  "An out-of-tree ConstraintGen" should "verify" in
    assert(WidthOp(Value(IntegerType(IntData(32), Signless))).verify().isOK)

  it should "report its own diagnostic" in {
    val r = WidthOp(Value(IntegerType(IntData(64), Signless))).verify()
    assert(r.isError)
    assert(
      r.getError.msg == "operand 'x': Expected an integer of width 32, got i64"
    )
  }

  it should "reject an attribute of the wrong shape" in
    assert(WidthOp(Value(Float32Type())).verify().isError)

  it should "take part in result type inference" in {
    val ctx = MLContext()
    ctx.registerDialect(ExtDialect)
    val parser = new Parser(ctx, allowUnregisteredDialect = false)
    parser.parse(
      """%1 = ext.width_infer %0 : i64""",
      operationP(using _, parser),
    ) match
      case fastparse.Parsed.Success(op, _) =>
        assert(op.results.head.typ == IntegerType(IntData(64), Signless))
      case f: fastparse.Parsed.Failure => fail(s"parse failed: ${f.msg}")
  }

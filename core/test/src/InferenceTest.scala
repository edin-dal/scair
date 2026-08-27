package scair.constrainttest

import org.scalatest.flatspec.*
import scair.*
import scair.clair.*
import scair.clair.macros.*
import scair.constraints.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.parse.*

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||   OPS WITH AN ELIDED RESULT TYPE  ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

/** MLIR's `SameOperandsAndResultType` idiom: the format spells out the operand
  * type only, and the result type follows from the shared constraint variable.
  */
case class SameTypeUnary(
    x: Operand[Attribute !> Var["T"]],
    res: Result[Attribute !> Var["T"]],
) extends DerivedOperation["infer.unary"]
    with AssemblyFormat["$x attr-dict `:` type($x)"] derives OpDefs

/** The result type is pinned outright, so the format need not mention it. */
case class FixedResult(
    x: Operand[Attribute !> Var["T"]],
    res: Result[Attribute !> EqAttr[f32.type]],
) extends DerivedOperation["infer.fixed"]
    with AssemblyFormat["$x attr-dict `:` type($x)"] derives OpDefs

/** The result is a whole attribute built out of a parameter inferred from a
  * parameter of the operand's type.
  */
case class BuiltResult(
    x: Operand[Attribute !> Param[IntegerType, (Var["W"], AnyAttr)]],
    res: Result[
      Attribute !> Param[IntegerType, (Var["W"], EqAttr[Signless.type])]
    ],
) extends DerivedOperation["infer.built"]
    with AssemblyFormat["$x attr-dict `:` type($x)"] derives OpDefs

val InferDialect =
  summonDialect[EmptyTuple, (SameTypeUnary, FixedResult, BuiltResult)]

class InferenceTest extends AnyFlatSpec:

  val ctx = MLContext()
  ctx.registerDialect(InferDialect)

  def parseOp(src: String): Operation =
    val parser = new Parser(ctx, allowUnregisteredDialect = false)
    parser.parse(src, operationP(using _, parser)) match
      case fastparse.Parsed.Success(op, _) => op
      case f: fastparse.Parsed.Failure     =>
        fail(s"Failed to parse operation: ${f.msg}")

  "A result type absent from the assembly format" should
    "be inferred from a constraint variable bound by an operand" in {
      val op = parseOp("""%1 = infer.unary %0 : f64""")
      assert(op.results.head.typ == Float64Type())
    }

  it should "be inferred from an EqAttr" in {
    val op = parseOp("""%1 = infer.fixed %0 : f64""")
    assert(op.results.head.typ == Float32Type())
  }

  it should "be built from a parameter of an operand's type" in {
    val op = parseOp("""%1 = infer.built %0 : i32""")
    assert(op.results.head.typ == IntegerType(IntData(32), Signless))
  }

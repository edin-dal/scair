package scair

import fastparse.*
import org.scalatest.*
import org.scalatest.flatspec.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.clair.*
import scair.enums.I32Enum
import scair.parse.*

enum Color(name: String) extends I32Enum(name):
  case Red extends Color("red")
  case Green extends Color("green")
  case Blue extends Color("blue")

case class EnumOperation(
    val color: Color
) extends DerivedOperation["enum.enum_op"] derives OpDefs

// A standalone enum attribute: each case is an attribute in its own right,
// spelled `#enum.shade<pale>`, rather than being backed by an integer one.
enum Shade(caseName: String) extends EnumAttr("enum.shade", caseName):
  case Pale extends Shade("pale")
  case Vivid extends Shade("vivid")
  case Dark extends Shade("dark")

case class ShadeOperation(
    val shade: Shade,
    val accent: Option[Shade] = None,
) extends DerivedOperation["enum.shade_op"] derives OpDefs

val EnumTestDialect = summonDialect[
  Tuple1[Shade],
  (EnumOperation, ShadeOperation),
]

class EnumAttrTest extends AnyFlatSpec with BeforeAndAfter:

  val ctx = MLContext()
  ctx.registerDialect(EnumTestDialect)
  var parser = new Parser(ctx, allowUnregisteredDialect = false)

  before {
    parser = new Parser(ctx, allowUnregisteredDialect = false)
  }

  "EnumAttr" should "print and parse correctly" in {
    val parsed = parser.parse(
      """"enum.enum_op"() <{color = 0 : i32}> : () -> ()""",
      operationP(using _, parser),
    )
    parsed match
      case fastparse.Parsed.Success(value, _) =>
        val enumOp = value.asInstanceOf[EnumOperation]
        assert(enumOp.color == Color.Red)
      case failure: fastparse.Parsed.Failure =>
        fail(s"Failed to parse operation: $failure.msg")
  }

  it should "destructure and structure correctly" in {
    val op = EnumOperation(Color.Green)
    val destructured = summon[OpDefs[EnumOperation]].destructure(op)
    val restructured = summon[OpDefs[EnumOperation]].structure(destructured)
    assert(op.color == restructured.color)
  }

  "EnumAttr" should "print as #dialect.enum<case>" in {
    assert(Shade.Pale.toString == "#enum.shade<pale>")
    assert(Shade.Dark.name == "enum.shade")
  }

  it should "parse every case back to its own instance" in {
    for shade <- Shade.values do
      val parsed = parser.parse(
        s""""enum.shade_op"() <{shade = $shade}> : () -> ()""",
        operationP(using _, parser),
      )
      parsed match
        case fastparse.Parsed.Success(value, _) =>
          assert(value.asInstanceOf[ShadeOperation].shade == shade)
        case failure: fastparse.Parsed.Failure =>
          fail(s"Failed to parse operation: ${failure.msg}")
  }

  it should "parse an optional case, present and absent" in {
    val withAccent = parser.parse(
      """"enum.shade_op"() <{shade = #enum.shade<pale>, accent = #enum.shade<dark>}> : () -> ()""",
      operationP(using _, parser),
    )
    withAccent match
      case fastparse.Parsed.Success(value, _) =>
        assert(value.asInstanceOf[ShadeOperation].accent == Some(Shade.Dark))
      case failure: fastparse.Parsed.Failure =>
        fail(s"Failed to parse operation: ${failure.msg}")

    val withoutAccent = parser.parse(
      """"enum.shade_op"() <{shade = #enum.shade<pale>}> : () -> ()""",
      operationP(using _, parser),
    )
    withoutAccent match
      case fastparse.Parsed.Success(value, _) =>
        assert(value.asInstanceOf[ShadeOperation].accent == None)
      case failure: fastparse.Parsed.Failure =>
        fail(s"Failed to parse operation: ${failure.msg}")
  }

  it should "reject an unknown case, listing the valid ones" in {
    val parsed = parser.parse(
      """"enum.shade_op"() <{shade = #enum.shade<garish>}> : () -> ()""",
      operationP(using _, parser),
    )
    parsed match
      case fastparse.Parsed.Success(value, _) =>
        fail(s"Expected a parse failure, got: $value")
      case failure: fastparse.Parsed.Failure =>
        assert(
          failure.trace().longMsg.contains(
            "expected one of pale, vivid, dark for enum enum.shade, but got 'garish'"
          )
        )
  }

  it should "destructure and structure standalone enums correctly" in {
    val op = ShadeOperation(Shade.Vivid, Some(Shade.Pale))
    val defs = summon[OpDefs[ShadeOperation]]
    val restructured = defs.structure(defs.destructure(op))
    assert(restructured.shade == Shade.Vivid)
    assert(restructured.accent == Some(Shade.Pale))

    val bare = ShadeOperation(Shade.Dark)
    val restructuredBare = defs.structure(defs.destructure(bare))
    assert(restructuredBare.shade == Shade.Dark)
    assert(restructuredBare.accent == None)
  }

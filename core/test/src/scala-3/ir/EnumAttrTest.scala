package scair

import fastparse.*
import org.scalatest.*
import org.scalatest.flatspec.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.clair.macros.*
import scair.enums.enumattr.I32Enum

enum Color(name: String) extends I32Enum(name):
  case Red extends Color("red")
  case Green extends Color("green")
  case Blue extends Color("blue")

case class EnumOperation(
    val color: Color
) extends DerivedOperation["enum.enum_op", EnumOperation]
    derives DerivedOperationCompanion

val EnumTestDialect = summonDialect[EmptyTuple, Tuple1[EnumOperation]]

class EnumAttrTest extends AnyFlatSpec with BeforeAndAfter:

  val ctx = MLContext()
  ctx.registerDialect(EnumTestDialect)
  var parser = new Parser(ctx, allowUnregisteredDialect = false)

  before {
    parser = new Parser(ctx, allowUnregisteredDialect = false)
  }

  "EnumAttr" should "print and parse correctly" in {
    val parsed = parser.parseThis(
      """"enum.enum_op"() <{color = 0 : i32}> : () -> ()""",
      parser.GenericOperation(Seq())(using _)
    )
    parsed match
      case fastparse.Parsed.Success(value, _) =>
        val enumOp = value.asInstanceOf[EnumOperation]
        assert(enumOp.color == Color.Red)
      case failure: fastparse.Parsed.Failure =>
        fail(s"Failed to parse operation: $failure.msg")
  }

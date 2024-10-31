package scair

import org.scalatest._
import Inspectors._
import flatspec._
import matchers.should.Matchers._
import prop._
import Tables._
import exceptions._

import fastparse._, MultiLineWhitespace._
import scala.collection.mutable
import scala.util.{Try, Success, Failure}
import clair._

import scair.dialects.builtin._

class ClairParserTests
    extends AnyFlatSpec
    with BeforeAndAfter
    with TableDrivenPropertyChecks {

  def xor[A](parsed: Parsed[Any], result: String): String =
    parsed match {
      case Parsed.Success(expected, x) =>
        if (result == "Succeed") then "pass" else expected.toString
      case Parsed.Failure(msg, _, _) =>
        if (result == "Fail") then "pass" else msg
    }

  var parser: ClairParser = new ClairParser
  var ctx: ParseCTX = new ParseCTX

  before {
    parser = new ClairParser
    ctx = new ParseCTX
    ctx.addCTXtype("a", RegularType("dialect1", "i1"))
    ctx.addCTXtype("b", RegularType("dialect1", "i2"))
    ctx.addCTXtype("c", RegularType("dialect1", "i3"))
  }

  val typeTests = Table(
    ("input", "result"),
    ("{ type tt = dialect.name }", "Succeed"),
    ("{ type tt == dialect.name }", "Fail")
  )

  val constraintValTests = Table(
    ("input", "result"),
    ("map : a | b | c", "Succeed"),
    ("map = a | b | c", "Fail"),
    ("map == a", "Succeed"),
    ("map = a", "Fail"),
    ("map : a", "Succeed"),
    ("map :: a", "Fail"),
    ("map - a", "Fail")
  )

  val constraintDictTests = Table(
    ("input", "result"),
    ("map : a | b | c", "Fail"),
    ("map = a | b | c", "Succeed"),
    ("map == a", "Succeed"),
    ("map = a", "Succeed"),
    ("map : a", "Fail"),
    ("map :: a", "Fail"),
    ("map - a", "Fail")
  )

  val attrTypeTest = Table(
    ("input", "result"),
    ("-> type", "Succeed"),
    ("-> tesa", "Fail")
  )

  val attrDataTest = Table(
    ("input", "result"),
    ("-> data", "Succeed"),
    ("-> dsad", "Fail")
  )

  val opOperandsTest = Table(
    ("input", "result"),
    ("-> operands [map:a, map1:b, map2:c]", "Succeed"),
    ("-> operands [map=a, map1=b, map2=c]", "Fail"),
    ("-> operands []", "Fail"),
    ("-> operand [map:a, map1:b, map2:c]", "Fail")
  )

  val opResultsTest = Table(
    ("input", "result"),
    ("-> results [map:a, map1:b, map2:c]", "Succeed"),
    ("-> results [map=a, map1=b, map2=c]", "Fail"),
    ("-> results []", "Fail"),
    ("-> result [map:a, map1:b, map2:c]", "Fail")
  )

  val opRegionsTest = Table(
    ("input", "result"),
    ("-> regions [1]", "Succeed"),
    ("-> regions [1.0]", "Fail"),
    ("-> regions []", "Fail")
  )

  val opSuccessorsTest = Table(
    ("input", "result"),
    ("-> successors [1]", "Succeed"),
    ("-> successors [1.0]", "Fail"),
    ("-> successors []", "Fail")
  )

  val opPropertiesTest = Table(
    ("input", "result"),
    ("-> properties [map=a, map1=b, map2=c]", "Succeed"),
    ("-> properties [map:a, map1:b, map2:c]", "Fail"),
    ("-> properties []", "Fail"),
    ("-> propertie [map=a, map1=b, map2=c]", "Fail")
  )

  val opAttributesTest = Table(
    ("input", "result"),
    ("-> attributes [map=a, map1=b, map2=c]", "Succeed"),
    ("-> attributes [map:a, map1:b, map2:c]", "Fail"),
    ("-> attributes []", "Fail"),
    ("-> attribute [map=a, map1=b, map2=c]", "Fail")
  )

  val unitTests = Table(
    ("name", "pattern", "tests"),
    (
      "Type Test",
      ((x: fastparse.P[_]) => ClairParser.RegularTypeP(x, ctx)),
      typeTests
    ),
    (
      "Constraints Value Test",
      ((x: fastparse.P[_]) => ClairParser.ValueDef(x, ctx)),
      constraintValTests
    ),
    (
      "Constraints Dict Test",
      ((x: fastparse.P[_]) => ClairParser.DictDef(x, ctx)),
      constraintDictTests
    ),
    (
      "Attribute type Test",
      ((x: fastparse.P[_]) => ClairParser.TypeInput(x)),
      attrTypeTest
    ),
    (
      "Attribute data Test",
      ((x: fastparse.P[_]) => ClairParser.DataInput(x)),
      attrDataTest
    ),
    (
      "Operation Operand Test",
      ((x: fastparse.P[_]) => ClairParser.OperandsInput(x, ctx)),
      opOperandsTest
    ),
    (
      "Operation Result Test",
      ((x: fastparse.P[_]) => ClairParser.ResultsInput(x, ctx)),
      opResultsTest
    ),
    (
      "Operation Region Test",
      ((x: fastparse.P[_]) => ClairParser.RegionsInput(x)),
      opRegionsTest
    ),
    (
      "Operation Successor Test",
      ((x: fastparse.P[_]) => ClairParser.SuccessorsInput(x)),
      opSuccessorsTest
    ),
    (
      "Operation Properties Test",
      ((x: fastparse.P[_]) => ClairParser.OpPropertiesInput(x, ctx)),
      opPropertiesTest
    ),
    (
      "Operation Attributes Test",
      ((x: fastparse.P[_]) => ClairParser.OpAttributesInput(x, ctx)),
      opAttributesTest
    )
  )

  forAll(unitTests) { (name, pattern, tests) =>
    forAll(tests) { (input, result) =>
      s"$name [$input]" should s"[$result]" in {
        xor(parser.parseThis(input, pattern), result) shouldBe "pass"
      }
    }
  }
}

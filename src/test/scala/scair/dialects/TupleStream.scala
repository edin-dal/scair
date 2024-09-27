package scair

import org.scalatest._
import flatspec._
import matchers.should.Matchers._
import prop._

import fastparse._, MultiLineWhitespace._
import scala.collection.mutable
import scala.util.{Try, Success, Failure}
import Parser._
import org.scalatest.prop.Tables.Table
import org.scalatest.prop.TableDrivenPropertyChecks.forAll
import AttrParser._

import scala.collection.mutable.ArrayBuffer

import scair.dialects.builtin._
import scair.dialects.LingoDB.TupleStream._

class TupleStreamTest extends AnyFlatSpec with BeforeAndAfter {

  var printer = new Printer(true)

  before {
    printer = new Printer(true)
  }

  def getResult[A](result: String, expected: A) =
    result match {
      case "Success" => ((x: Int) => Parsed.Success(expected, x))
      case "Failure" => Parsed.Failure(_, _, _)
    }

  val strToAttributeTests =
    Table(
      ("input", "result", "expected"),
      (
        "!tuples.tuple<f32, f32>",
        "Success",
        TupleStreamTuple(Seq(Float32Type, Float32Type))
      ),
      (
        "!tuples.tuplestream<f32, f32>",
        "Success",
        TupleStreamTuple(Seq(Float32Type, Float32Type))
      )
    )

  forAll(strToAttributeTests) { (input, result, expected) =>
    // Get the expected output
    val res = getResult(result, expected)
    "strToAttributeTests" should s"[ '$input' -> '$expected' = $result ]" in {
      // Run the pqrser on the input and check
      parse(input, Type(_)) should matchPattern { case res => // pass
      }
    }
  }
}

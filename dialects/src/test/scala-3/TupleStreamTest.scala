package scair

import fastparse.*
import org.scalatest.*
import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import org.scalatest.prop.*
import org.scalatest.prop.TableDrivenPropertyChecks.forAll
import org.scalatest.prop.Tables.Table
import scair.dialects.LingoDB.TupleStream.*
import scair.dialects.builtin.*

class TupleStreamTest extends AnyFlatSpec with BeforeAndAfter {
  val ctx = new MLContext()
  ctx.registerDialect(TupleStreamDialect)
  var parser = new Parser(ctx)
  var printer = new Printer(true)

  before {
    parser = new Parser(ctx)
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
      parse(input, parser.Type(_)) should matchPattern { case res => // pass
      }
    }
  }
}

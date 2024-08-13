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
import scair.dialects.CMath.cmath._

class CMathTest extends AnyFlatSpec with BeforeAndAfter {

  var printer = new Printer

  before {
    printer = new Printer
  }

  def getResult[A](result: String, expected: A) =
    result match {
      case "Success" => ((x: Int) => Parsed.Success(expected, x))
      case "Failure" => Parsed.Failure(_, _, _)
    }

  val strToAttributeTests =
    Table(
      ("input", "result", "expected"),
      ("!cmath.complex<f32>", "Success", ComplexType(Seq(Float32Type)))
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

  "Cmath simple object creation test :)" should "match parsed string against expected string" in {

    var parser: Parser = new Parser

    val input = """"op1"() ({
                   |^bb0(%1 : f32, %3 : f32, %4 : f32):
                   |    %0 = "cmath.norm"(%1) : (f32) -> (f64)
                   |    %2 = "cmath.mul"(%3, %4) : (f32, f32) -> (f32)
                   |}) : () -> ()""".stripMargin

    parser.parseThis(
      text = input,
      pattern = parser.TopLevel(_)
    ) should matchPattern {
      case Parsed.Success(
            ModuleOp(
              ArrayBuffer(),
              ArrayBuffer(),
              Seq(),
              Seq(
                Region(
                  Seq(
                    Block(
                      Seq(
                        UnregisteredOperation(
                          "op1",
                          ArrayBuffer(),
                          ArrayBuffer(),
                          Seq(),
                          Seq(
                            Region(
                              Seq(
                                Block(
                                  Seq(
                                    Norm(
                                      ArrayBuffer(Value(Float32Type)),
                                      ArrayBuffer(),
                                      Seq(Value(Float64Type)),
                                      Seq(),
                                      _,
                                      _
                                    ),
                                    Mul(
                                      ArrayBuffer(
                                        Value(Float32Type),
                                        Value(Float32Type)
                                      ),
                                      ArrayBuffer(),
                                      Seq(Value(Float32Type)),
                                      Seq(),
                                      _,
                                      _
                                    )
                                  ),
                                  Seq(
                                    Value(Float32Type),
                                    Value(Float32Type),
                                    Value(Float32Type)
                                  )
                                )
                              )
                            )
                          ),
                          _,
                          _
                        )
                      ),
                      Seq()
                    )
                  )
                )
              ),
              _,
              _
            ),
            _
          ) =>
    }
  }
}

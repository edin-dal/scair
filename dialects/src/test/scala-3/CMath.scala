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

import scair.dialects.builtin._
import scair.dialects.CMath.cmath._
import scair.ir._

class CMathTest extends AnyFlatSpec with BeforeAndAfter {

  val ctx = new MLContext()
  ctx.registerDialect(CMath)

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
      ("!cmath.complex<f32>", "Success", ComplexType(Seq(Float32Type)))
    )

  forAll(strToAttributeTests) { (input, result, expected) =>
    // Get the expected output
    val res = getResult(result, expected)
    "strToAttributeTests" should s"[ '$input' -> '$expected' = $result ]" in {
      // Run the parser on the input and check
      parse(input, parser.Type(_)) should matchPattern { case res => // pass
      }
    }
  }

  "Cmath simple object creation test :)" should "match parsed string against expected string" in {

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
              ListType(),
              ListType(),
              ListType(),
              ListType(
                Region(
                  Seq(
                    Block(
                      ListType(
                        UnregisteredOperation(
                          "op1",
                          ListType(),
                          ListType(),
                          ListType(),
                          ListType(
                            Region(
                              Seq(
                                Block(
                                  ListType(
                                    Norm(
                                      ListType(Value(Float32Type)),
                                      ListType(),
                                      ListType(Value(Float64Type)),
                                      ListType(),
                                      _,
                                      _
                                    ),
                                    Mul(
                                      ListType(
                                        Value(Float32Type),
                                        Value(Float32Type)
                                      ),
                                      ListType(),
                                      ListType(Value(Float32Type)),
                                      ListType(),
                                      _,
                                      _
                                    )
                                  ),
                                  ListType(
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
                      ListType()
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

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
import scair.dialects.cmath._

class CMathTest extends AnyFlatSpec with BeforeAndAfter {
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
            Seq(
              UnregisteredOperation(
                "op1",
                Seq(),
                ArrayBuffer(),
                Seq(),
                Seq(
                  Region(
                    Seq(
                      Block(
                        Seq(
                          Norm(
                            Seq(Value(Float32Type)),
                            ArrayBuffer(),
                            Seq(Value(Float64Type)),
                            Seq(),
                            _,
                            _
                          ),
                          Mul(
                            Seq(Value(Float32Type), Value(Float32Type)),
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
                    ),
                    _
                  )
                ),
                _,
                _
              )
            ),
            _
          ) =>
    }
  }
}

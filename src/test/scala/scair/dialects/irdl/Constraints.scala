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

import AttrParser._
import Parser._
import scair.dialects.builtin._
import scair.dialects.irdl._

class ConstraintsTest extends AnyFlatSpec with BeforeAndAfter {

  var constraint_ctx = new ConstraintContext()

  before {
    constraint_ctx = new ConstraintContext()
  }

  "Any Constraint Test" should "test AnyAttr constraint" in {
    val any_cnstrt = AnyAttr
    any_cnstrt.verify(Float16Type, constraint_ctx)
  }

  // "Equal Constraint Test" should "test EqualAttr constraint" in {

  //   val this_cnstrt =
  //   val that_cnstrt =

  //   any_cnstrt.verify(Float16Type)
  // }

  // "Region2 - Unit Tests" should "parse correctly" in {

  //   withClue("Test 2: ") {
  //     val exception = intercept[Exception](
  //       parser.parseThis(
  //         text =
  //           "{^bb0(%5: i32):\n" + "%0, %1, %2 = \"test.op\"() : () -> (i32, i64, i32)\n" +
  //             "\"test.op\"(%1, %0) : (i64, i32) -> ()" + "^bb0(%4: i32):\n" + "%7, %8, %9 = \"test.op\"() : () -> (i32, i64, i32)\n" +
  //             "\"test.op\"(%8, %7) : (i64, i32) -> ()" + "}",
  //         pattern = parser.Region(_)
  //       )
  //     ).getMessage shouldBe "Block cannot be defined twice within the same scope - ^bb0"
  //   }
  // }

}

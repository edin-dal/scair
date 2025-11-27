package scair.ir

import org.scalatest.*
import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*

import org.scalatest.prop.TableDrivenPropertyChecks.forAll
import org.scalatest.prop.Tables.Table
import scair.Printer
import scair.dialects.builtin.*
import java.io.StringWriter
import java.io.PrintWriter
import scair.clair.macros.*
import scala.collection.mutable.LinkedHashMap

import scair.dialects.test.TestOp

class DeepCopyTest extends AnyFlatSpec:

  "Operation.deepCopy" should "deep copy a simple operation" in:
    val a = TestOp(
      properties = Map("prop" -> I32),
      attributes = LinkedHashMap("attr" -> I64),
      results = Seq(Result(I32))
    )
    val b = a.deepCopy
    a should not be b
    a.attributes.equals(b.attributes) shouldBe true
    a.attributes.eq(b.attributes) shouldBe false
    a.properties.equals(b.properties) shouldBe true
    a should matchPattern {
      case TestOp(
            operands = Seq(),
            successors = Seq(),
            results = Seq(Result(I32)),
            regions = Seq()
          ) =>
        ()
    }

  it should "deep copy use-def operations" in:
    val value = Result(I32)
    val prod = TestOp(results = Seq(value))
    val user = TestOp(operands = Seq(value))

    val a = Block(operations = Seq(prod, user))
    val b = a.deepCopy

    a should not be b
    b should matchPattern {
      case Block(
            operations = BlockOperations(
              TestOp(results = Seq(u)),
              TestOp(operands = Seq(v))
            )
          ) if (u eq v) && !(v eq value) =>
        ()
    }

  it should "deep copy nested operations" in:
    // Children
    val ca0 = TestOp(results = Seq(Result(I32)))
    val ca1 = TestOp(results = Seq(Result(I32)))
    // Parent
    val pa = TestOp(
      regions = Seq(
        Region(Block(arguments_types = Seq(I32), operations = Seq(ca0))),
        Region(Block(arguments_types = Seq(I32), operations = Seq(ca1)))
      )
    )

    val pb = pa.deepCopy

    pa should not be pb
    pb should matchPattern {
      case TestOp(
            regions = Seq(
              Region(
                Block(
                  operations = BlockOperations(
                    a @ TestOp(results = Seq(Result(I32)))
                  )
                )
              ),
              Region(
                Block(
                  operations = BlockOperations(
                    b @ TestOp(results = Seq(Result(I32)))
                  )
                )
              )
            )
          )
          if !(a eq ca0) && !(b eq ca1) && !(a.results.head eq ca0.results.head) && !(b.results.head eq ca1.results.head) =>
        ()
    }

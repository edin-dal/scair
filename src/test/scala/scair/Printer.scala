package scair

import org.scalatest._
import Inspectors._
import flatspec._
import matchers.should.Matchers._
import prop._

import scala.collection.mutable
import Parser._

import scair.dialects.builtin._

class PrinterTest extends AnyFlatSpec with BeforeAndAfter {

  var printer = new Printer();

  before {
    printer = new Printer
  }

  val F32 = Float32Type
  val F64 = Float64Type

  val I32 = new IntegerType(32, Signless)
  val I64 = new IntegerType(64, Signless)

  "printRegion" should "return the correct string representation of a region" in {
    val region =
      Region(
        Seq(Block(Seq(UnregisteredOperation("op1"))))
      )
    val expected = """{
                     |^bb0():
                     |  "op1"() : () -> ()
                     |}""".stripMargin
    val result = printer.printRegion(region)
    result shouldEqual expected
  }

  "printBlock" should "return the correct string representation of a block" in {
    val block = Block(
      Seq(UnregisteredOperation("op1")),
      Seq(Value(I32))
    )
    val expected = """^bb0(%0: i32):
                     |  "op1"() : () -> ()""".stripMargin
    val result = printer.printBlock(block)
    result shouldEqual expected
  }

  "printValue" should "return the correct string representation of a value" in {
    val value = Value(I32)
    "%0" shouldEqual printer.printValue(value)
    "i32" shouldEqual printer.printAttribute(value.typ)
    "%0: i32" shouldEqual printer.printBlockArgument(value)
  }

  "printOperation" should "return the correct string representation of an operation" in {
    val operation = UnregisteredOperation(
      "op1",
      results = Seq(new Value(F32)),
      operands = ListType(new Value(F32)),
      regions = Seq(
        Region(
          Seq(
            Block(
              Seq(UnregisteredOperation("op2"))
            )
          )
        )
      )
    )
    val expected = """%0 = "op1"(%1) ({
                     |^bb0():
                     |  "op2"() : () -> ()
                     |}) : (f32) -> (f32)""".stripMargin
    val result = printer.printOperation(operation)
    result shouldEqual expected
  }

  "printProgram" should "return the correct string representation of a program" in {
    val program = Seq(
      UnregisteredOperation("op1"),
      UnregisteredOperation("op2")
    )
    val expected = """"op1"() : () -> ()
                     |"op2"() : () -> ()""".stripMargin
    val result = printer.printProgram(program)
    // result shouldEqual expected
  }

  "printSuccessors" should "return the correct string representation of a operation with successors" in {

    val val1 = new Value(I32)
    val val2 = new Value(I64)
    val val3 = new Value(I32)
    val val4 = new Value(I64)

    val successorTestBlock = new Block(
      Seq(),
      Seq()
    )

    val program =
      UnregisteredOperation(
        "op1",
        regions = Seq(
          Region(
            Seq(
              successorTestBlock,
              Block(
                Seq(
                  UnregisteredOperation(
                    "test.op",
                    ListType(),
                    ListType(successorTestBlock),
                    Seq(),
                    Seq()
                  )
                ),
                Seq()
              )
            )
          )
        )
      )
    val expected = """"op1"() ({
                     |^bb0():
                     |
                     |^bb1():
                     |  "test.op"()[^bb0] : () -> ()
                     |}) : () -> ()""".stripMargin
    val result = printer.printOperation(program)
    result shouldEqual expected
  }
}

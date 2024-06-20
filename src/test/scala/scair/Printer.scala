package scair

import org.scalatest._
import Matchers._
import Parser._

class PrinterTest extends FlatSpec {

  "printRegion" should "return the correct string representation of a region" in {
    val region =
      Region(
        Seq(Block(Seq(Operation("op1", Seq(), Seq(), Seq(), Seq())), Seq()))
      )
    val expected = """{
                     |^bb0():
                     |  "op1"() : () -> ()
                     |}""".stripMargin
    val result = Printer.printRegion(region)
    result shouldEqual expected
  }

  "printBlock" should "return the correct string representation of a block" in {
    val block = Block(
      Seq(Operation("op1", Seq(), Seq(), Seq(), Seq())),
      Seq(Value(Attribute("i32")))
    )
    val expected = """^bb1(%0: i32):
                     |  "op1"() : () -> ()""".stripMargin
    val result = Printer.printBlock(block)
    result shouldEqual expected
  }

  "printAttribute" should "return the correct string representation of an attribute" in {
    val attribute = Attribute("name")
    val expected = "name"
    val result = Printer.printAttribute(attribute)
    result shouldEqual expected
  }

  "printValue" should "return the correct string representation of a value" in {
    val value = Value(Attribute("i32"))
    "%1" shouldEqual Printer.printValue(value)
    "i32" shouldEqual Printer.printAttribute(value.typ)
    "%1: i32" shouldEqual Printer.printBlockArgument(value)
  }

  "printOperation" should "return the correct string representation of an operation" in {
    val operation = Operation(
      "op1",
      Seq(Value(Attribute("i32"))),
      Seq(),
      Seq(Value(Attribute("i32"))),
      Seq(
        Region(
          Seq(Block(Seq(Operation("op2", Seq(), Seq(), Seq(), Seq())), Seq()))
        )
      )
    )
    val expected = """%2 = "op1"(%3)({
                     |  ^bb2():
                     |    "op2"() : () -> ()
                     |  }) : (i32) -> (i32)""".stripMargin
    val result = Printer.printOperation(operation)
    result shouldEqual expected
  }

  "printProgram" should "return the correct string representation of a program" in {
    val program = Seq(
      Operation("op1", Seq(), Seq(), Seq(), Seq()),
      Operation("op2", Seq(), Seq(), Seq(), Seq())
    )
    val expected = """"op1"() : () -> ()
                     |"op2"() : () -> ()""".stripMargin
    val result = Printer.printProgram(program)
    // result shouldEqual expected
  }
  "printSuccessors" should "return the correct string representation of a operation with successors" in {

    val val1 = Value(Attribute("i32"))
    val val2 = Value(Attribute("i64"))
    val val3 = Value(Attribute("i32"))
    val val4 = Value(Attribute("i64"))

    val successorTestBlock = Block(
      Seq(
        Operation(
          "test.op",
          Seq(),
          Seq(),
          Seq(
            val3,
            val4,
            Value(Attribute("i32"))
          ),
          Seq()
        ),
        Operation(
          "test.op",
          Seq(val4, val3),
          Seq(),
          Seq(),
          Seq()
        )
      ),
      Seq(Value(Attribute("i32")))
    )

    val program =
      Operation(
        "op1",
        Seq(),
        Seq(),
        Seq(),
        Seq(
          Region(
            Seq(
              successorTestBlock,
              Block(
                Seq(
                  Operation(
                    "test.op",
                    Seq(),
                    Seq(successorTestBlock),
                    Seq(
                      val1,
                      val2,
                      Value(Attribute("i32"))
                    ),
                    Seq()
                  ),
                  Operation(
                    "test.op",
                    Seq(
                      val2,
                      val1
                    ),
                    Seq(),
                    Seq(),
                    Seq()
                  )
                ),
                Seq(Value(Attribute("i32")))
              )
            ),
            None
          )
        )
      )
    val expected = """"op1"()({
                     |  ^bb3(%4: i32):
                     |    %5, %6, %7 = "test.op"() : () -> (i32, i64, i32)
                     |    "test.op"(%6, %5) : (i64, i32) -> ()
                     |  ^bb4(%8: i32):
                     |    %9, %10, %11 = "test.op"()[^bb3] : () -> (i32, i64, i32)
                     |    "test.op"(%10, %9) : (i64, i32) -> ()
                     |  }) : () -> ()""".stripMargin
    val result = Printer.printOperation(program)
    println(result)
    result shouldEqual expected
  }
}

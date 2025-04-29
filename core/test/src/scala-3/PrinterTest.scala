package scair

import org.scalatest.*
import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import scair.dialects.builtin.*
import scair.ir.*
import java.io.*

// class PrinterTest extends AnyFlatSpec with BeforeAndAfter {

//   var out = StringWriter()
//   var printer = new Printer(true, p = PrintWriter(out));

//   before {
//     printer = new Printer(true, p = PrintWriter(out))
//   }

//   val F32 = Float32Type
//   val F64 = Float64Type

//   val I32 = new IntegerType(IntData(32), Signless)
//   val I64 = new IntegerType(IntData(64), Signless)

//   "printRegion" should "return the correct string representation of a region" in {
//     val region =
//       Region(
//         Seq(Block(ListType(UnregisteredOperation("op1"))))
//       )
//     val expected = """{
//                      |  "op1"() : () -> ()
//                      |}""".stripMargin
//     printer.printRegion(region)
//     val result = out.toString()
//     result shouldEqual expected
//   }

//   "printBlock" should "return the correct string representation of a block" in {
//     val block = Block(
//       Seq(I32),
//       Seq(UnregisteredOperation("op1"))
//     )
//     val expected = """^bb0(%0: i32):
//                      |  "op1"() : () -> ()
//                      |""".stripMargin
//     printer.printBlock(block)
//     val result = out.toString()
//     result shouldEqual expected
//   }

//   "printValue" should "return the correct string representation of a value - 1" in {
//     val value = Value(I32)
//     printer.printValue(value)
//     "%0" shouldEqual out.toString()
//   }
//   "printValue" should "return the correct string representation of a value - 2" in {
//     val value = Value(I32)
//     printer.printValue(value)
//     "i32" shouldEqual printer.printAttribute(value.typ)
//   }
//   "printValue" should "return the correct string representation of a value - 3" in {
//     val value = Value(I32)
//     printer.printValue(value)
//     "%0: i32" shouldEqual printer.printBlockArgument(value)
//   }

//   "printOperation" should "return the correct string representation of an operation" in {
//     val operation = UnregisteredOperation(
//       "op1",
//       results_types = Seq(F32),
//       operands = Seq(Value(F32)),
//       regions = Seq(
//         Region(
//           Seq(
//             Block(
//               ListType(UnregisteredOperation("op2"))
//             )
//           )
//         )
//       )
//     )
//     val expected = """%0 = "op1"(%1) ({
//                      |  "op2"() : () -> ()
//                      |}) : (f32) -> (f32)""".stripMargin
//     val result = printer.printOperation(operation)
//     result shouldEqual expected
//   }

//   "printSuccessors" should "return the correct string representation of a operation with successors" in {

//     val val1 = Value(I32)
//     val val2 = Value(I64)
//     val val3 = Value(I32)
//     val val4 = Value(I64)

//     val successorTestBlock = Block(
//       ListType(),
//       ListType()
//     )

//     val program =
//       UnregisteredOperation(
//         "op1",
//         regions = Seq(
//           Region(
//             Seq(
//               successorTestBlock,
//               Block(
//                 Seq(
//                   UnregisteredOperation(
//                     "test.op",
//                     successors = Seq(successorTestBlock)
//                   )
//                 )
//               )
//             )
//           )
//         )
//       )
//     val expected = """"op1"() ({
//                      |^bb0():
//                      |
//                      |^bb1():
//                      |  "test.op"()[^bb0] : () -> ()
//                      |}) : () -> ()""".stripMargin
//     val result = printer.printOperation(program)
//     result shouldEqual expected
//   }

// }

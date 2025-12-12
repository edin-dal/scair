package scair

import fastparse.*
import org.scalatest.*
import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import org.scalatest.prop.*
import scair.dialects.builtin.*
import scair.dialects.cmath.*
import scair.parse.*

class TransformationsTest
    extends AnyFlatSpec
    with BeforeAndAfter
    with TableDrivenPropertyChecks:

  val I32 = IntegerType(IntData(32), Signless)
  val I64 = IntegerType(IntData(64), Signless)

//   def getResult[A](result: String, expected: A) =
//     result match {
//       case "Success" => ((x: Int) => Parsed.Success(expected, x))
//       case "Failure" => Parsed.Failure(_, _, _)
//     }
  val ctx = new MLContext()
  ctx.registerDialect(cmath)

  var parser: Parser = new Parser(ctx, allowUnregisteredDialect = true)
  var printer = new Printer(true)

  before {
    parser = new Parser(ctx, allowUnregisteredDialect = true)
    printer = new Printer(true)
  }

  "Operation Erasure" should
    "Test that operation does not get erased and throws error :)" in
    withClue("Operand Erasure: ") {

      val text = """  %0, %1 = "test.op"() : () -> (i32, i64)
                    | %2 = "test.op"(%0) : (i32) -> i32
                    | "op1"(%0, %1, %2) : (i32, i64, i32) -> ()
                    | "op2"(%0, %1, %2) : (i32, i64, i32) -> ()
                    | "op3"(%0, %1, %2) : (i32, i64, i32) -> ()
                    | "op4"(%0, %1, %2) : (i32, i64, i32) -> ()""".stripMargin

      val Parsed.Success(value, _) = parser.parse(
        input = text,
        parser = topLevelP(using _, parser),
      ): @unchecked

      val opToErase = value.regions(0).blocks(0).operations(1)

      val block =
        opToErase.containerBlock.getOrElse(throw new Exception("bruh"))

      val exception = intercept[Exception](
        block.eraseOp(opToErase)
      ).getMessage shouldBe
        "Attempting to erase a Value that has uses in other operations."

      opToErase.containerBlock shouldEqual None
    }

  "Operation Adding" should
    "Test that adding op to it's own block does not work :)" in
    withClue("Operand Adding: ") {

      val text = """  
        | %0, %1, %2 = "test.op"() ({
        | ^bb0(%3 : f32, %4 : f32, %5 : f32):
        |   "test.op"() ({
        |       ^bb1(%6 : f32, %7 : f32, %8 : f32):
        |           "test.op"()  : () -> ()
        |           %9 = "cmath.norm"(%6) : (f32) -> f64
        |           %10 = "cmath.mul"(%7, %8) : (f32, f32) -> f32
        |       }) {attr = "this is it"} : () -> ()
        |   %6 = "cmath.norm"(%3) : (f32) -> f64
        |   %7 = "cmath.mul"(%4, %5) : (f32, f32) -> f32
        | }) : () -> (!cmath.complex<f32>, !cmath.complex<index>, !cmath.complex<f32>)"""
        .stripMargin

      val Parsed.Success(value, _) = parser.parse(
        input = text,
        parser = topLevelP(using _, parser),
      ): @unchecked

      val opToAdd = value.regions(0).blocks(0).operations(0)

      val blockToAddItTo =
        opToAdd.regions(0).blocks(0).operations(0).regions(0).blocks(0)

      value.regions(0).blocks(0).detachOp(opToAdd)

      val exception = intercept[Exception](
        blockToAddItTo.addOp(opToAdd)
      ).getMessage shouldBe
        "Can't add an operation to a block that is contained within that operation"
    }

  "Operation Insertion" should
    "Test that operation does not get inserted into the wrong block and throws error :)" in
    withClue("Operand Insertion: ") {

      val text = """  %0, %1 = "test.op"() : () -> (i32, i64)
                    | %2 = "test.op"(%0) : (i32) -> i32
                    | "test.op"() ({
                    |    ^bb0(%444 : f32, %555 : f32, %6666 : f32):
                    |        "test.op"()  : () -> ()
                    |        %3 = "cmath.norm"(%1) : (i64) -> f64
                    |        %4 = "cmath.mul"(%3, %2) : (f64, i32) -> f32
                    |    }) {attr = "this is it"} : () -> ()
                    | "op1"(%0, %1, %2) : (i32, i64, i32) -> ()
                    | "op2"(%0, %1, %2) : (i32, i64, i32) -> ()
                    | "op3"(%0, %1, %2) : (i32, i64, i32) -> ()
                    | "op4"(%0, %1, %2) : (i32, i64, i32) -> ()""".stripMargin

      val Parsed.Success(value, _) = parser.parse(
        input = text,
        parser = topLevelP(using _, parser),
      ): @unchecked

      val block =
        value.regions(0).blocks(0).operations(2).regions(0).blocks(0)

      val refOp = value.regions(0).blocks(0).operations(0)

      val opToInsert = value.regions(0).blocks(0).operations(2).regions(0)
        .blocks(0).operations(0)

      val exception = intercept[Exception](
        block.insertOpBefore(refOp, opToInsert)
      ).getMessage shouldBe
        "Can't insert the new operation into the block, as the operation that was " +
        "given as a point of reference does not exist in the current block."
    }

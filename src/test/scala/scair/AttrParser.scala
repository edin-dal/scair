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

class AttrParserTest extends AnyFlatSpec with BeforeAndAfter {

  var printer = new Printer

  before {
    printer = new Printer
  }

  def getResult[A](result: String, expected: A) =
    result match {
      case "Success" => ((x: Int) => Parsed.Success(expected, x))
      case "Failure" => Parsed.Failure(_, _, _)
    }

  val F16 = Float16Type
  val F32 = Float32Type
  val F64 = Float64Type
  val F80 = Float80Type
  val F128 = Float128Type
  val I1 = IntegerType(1, Signless)
  val I16 = IntegerType(16, Signless)
  val I32 = IntegerType(32, Signless)
  val I64 = IntegerType(64, Signless)
  val INDEX = IndexType

  val attrToStringTests = Table(
    ("input", "result", "expected"),
    (F16, "Success", "f16"),
    (F32, "Success", "f32"),
    (F64, "Success", "f64"),
    (F80, "Success", "f80"),
    (F128, "Success", "f128"),
    (I1, "Success", "i1"),
    (I16, "Success", "i16"),
    (I32, "Success", "i32"),
    (I64, "Success", "i64"),
    (IntegerType(874234232, Signless), "Success", "i874234232"),
    (IntegerType(7, Signed), "Success", "si7"),
    (IntegerType(8, Unsigned), "Success", "ui8"),
    (
      RankedTensorType(
        ArrayAttribute(Seq(IntAttr(3), IntAttr(-1), IntAttr(5))),
        Float32Type,
        None
      ),
      "Success",
      "tensor<3x?x5xf32>"
    ),
    (UnrankedTensorType(Float32Type), "Success", "tensor<*xf32>"),
    (
      ArrayAttribute(Seq(F64, ArrayAttribute(Seq()), StringAttribute("hello"))),
      "Success",
      "[f64, [], \"hello\"]"
    ),
    (StringAttribute("hello world!"), "Success", "\"hello world!\""),
    (INDEX, "Success", "index")
  )

  val strToAttributeTests = Table(
    ("input", "result", "expected"),
    ("f16", "Success", F16),
    ("f32", "Success", F32),
    ("f64", "Success", F64),
    ("f80", "Success", F80),
    ("f128", "Success", F128),
    ("i1", "Success", I1),
    ("i16", "Success", I16),
    ("i32", "Success", I32),
    ("i64", "Success", I64),
    ("i874234232", "Success", IntegerType(874234232, Signless)),
    ("874234232", "Success", IntegerType(874234232, Signless)),
    ("si7", "Success", IntegerType(7, Signed)),
    ("ui8", "Success", IntegerType(8, Unsigned)),
    ("index", "Success", INDEX),
    (
      "[f64, [], \"hello\"]",
      "Success",
      ArrayAttribute(Seq(F64, ArrayAttribute(Seq()), StringAttribute("hello")))
    ),
    ("\"hello world!\"", "Success", StringAttribute("hello world!")),
    (
      "tensor<3x?x5xf32>",
      "Success",
      RankedTensorType(
        ArrayAttribute(Seq(IntAttr(3), IntAttr(-1), IntAttr(5))),
        Float32Type,
        None
      )
    ),
    ("tensor<*xf32>", "Success", UnrankedTensorType(Float32Type)),
    ("fg12", "Failure", "")
  )

  forAll(attrToStringTests) { (input, result, expected) =>
    "attrToStringTests" should s"[ '$input' -> '$expected' = $result ]" in {
      // Run the pqrser on the input and check
      input.toString shouldEqual expected
    }
  }

  forAll(strToAttributeTests) { (input, result, expected) =>
    // Get the expected output
    val res = getResult(result, expected)
    "strToAttributeTests" should s"[ '$input' -> '$expected' = $result ]" in {
      // Run the pqrser on the input and check
      parse(input, AttrParser.BuiltIn(_)) should matchPattern {
        case res => // pass
      }
    }
  }

  val valF32 = Value[Float32Type.type](F32)
  val valF64 = Value[Float64Type.type](F64)
  val valI1 = Value[IntegerType](I1)
  val valI16 = Value[IntegerType](I16)
  val valINDEX = Value[IndexType.type](INDEX)

  "printAttributesWithinOp" should "return the correct string representation of a Operation with blocks and different attributes" in {

    val block1 = Block(
      Seq(
        UnregisteredOperation(
          "test.op",
          results = Seq(
            valF32,
            valF64,
            Value(F80)
          )
        ),
        UnregisteredOperation(
          "test.op",
          operands = ArrayBuffer(valF64, valF32)
        )
      ),
      Seq(Value(F128))
    )

    val block2 = Block(
      Seq(
        UnregisteredOperation(
          "test.op",
          successors = mutable.ArrayBuffer(block1),
          results = Seq(
            valI1,
            valI16,
            Value(I32)
          )
        ),
        UnregisteredOperation(
          "test.op",
          operands = ArrayBuffer(
            valI16,
            valI1
          )
        )
      ),
      Seq(Value(I32))
    )

    val block3 = Block(
      Seq(
        UnregisteredOperation(
          "test.op",
          results = Seq(
            valINDEX
          )
        ),
        UnregisteredOperation(
          "test.op",
          operands = ArrayBuffer(valINDEX)
        )
      ),
      Seq(Value(I64))
    )

    val program =
      UnregisteredOperation(
        "op1",
        regions = Seq(
          Region(
            Seq(
              block1,
              block2,
              block3
            ),
            None
          )
        )
      )

    val expected = """"op1"() ({
                     |^bb0(%0: f128):
                     |  %1, %2, %3 = "test.op"() : () -> (f32, f64, f80)
                     |  "test.op"(%2, %1) : (f64, f32) -> ()
                     |^bb1(%4: i32):
                     |  %5, %6, %7 = "test.op"()[^bb0] : () -> (i1, i16, i32)
                     |  "test.op"(%6, %5) : (i16, i1) -> ()
                     |^bb2(%8: i64):
                     |  %9 = "test.op"() : () -> (index)
                     |  "test.op"(%9) : (index) -> ()
                     |}) : () -> ()""".stripMargin
    val result = printer.printOperation(program)
    result shouldEqual expected
  }

  "parseDifferentAttributes" should "match parsed string against expected string" in {

    var parser: Parser = new Parser

    val block1 = new Block(
      Seq(
        UnregisteredOperation(
          "test.op",
          operands = ArrayBuffer(
            valF32,
            valF64,
            Value(F80)
          )
        ),
        UnregisteredOperation(
          "test.op",
          results = Seq(valF64, valF32)
        )
      ),
      Seq(Value(F16))
    )

    val block2 = new Block(
      Seq(
        UnregisteredOperation(
          "test.op",
          successors = mutable.ArrayBuffer(block1),
          results = Seq(
            valI1,
            valI16,
            Value(I32)
          )
        ),
        UnregisteredOperation(
          "test.op",
          operands = ArrayBuffer(
            valI16,
            valI1
          )
        )
      ),
      Seq(Value(F128))
    )

    val block3 = new Block(
      Seq(
        UnregisteredOperation(
          "test.op",
          results = Seq(
            valINDEX
          )
        ),
        UnregisteredOperation(
          "test.op",
          operands = ArrayBuffer(valINDEX)
        )
      ),
      Seq(Value(I64))
    )

    val program =
      UnregisteredOperation(
        "op1",
        regions = Seq(
          Region(
            Seq(
              block1,
              block2,
              block3
            ),
            None
          )
        )
      )

    val input = """"op1"()({
                     |  ^bb0(%0: f16):
                     |    %1, %2, %3 = "test.op"() : () -> (f32, f64, f80)
                     |    "test.op"(%2, %1) : (f64, f32) -> ()
                     |  ^bb1(%4: f128):
                     |    %5, %6, %7 = "test.op"()[^bb0] : () -> (i1, i16, i32)
                     |    "test.op"(%6, %5) : (i16, i1) -> ()
                     |  ^bb2(%8: i64):
                     |    %9 = "test.op"() : () -> (index)
                     |    "test.op"(%9) : (index) -> ()
                     |  }) : () -> ()""".stripMargin

    parser.parseThis(
      text = input,
      pattern = parser.TopLevel(_)
    ) should matchPattern { case Parsed.Success(program, _) => }
  }

  "parsingInteger" should "match parsed string against expected string" in {

    var parser: Parser = new Parser

    val input = """"op1"()({
                     |  ^bb0(%0: f128):
                     |    %1, %2, %3 = "test.op"() : () -> (i32, si64, ui80)
                     |  }) : () -> ()""".stripMargin

    parser
      .parseThis(
        text = input,
        pattern = parser.OperationPat(_)
      ) should matchPattern {
      case Parsed.Success(
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
                        UnregisteredOperation(
                          "test.op",
                          ArrayBuffer(),
                          ArrayBuffer(),
                          Seq(
                            Value(IntegerType(32, Signless)),
                            Value(IntegerType(64, Signed)),
                            Value(IntegerType(80, Unsigned))
                          ),
                          Seq(),
                          _,
                          _
                        )
                      ),
                      Seq(Value(F128))
                    )
                  ),
                  None
                )
              ),
              _,
              _
            ),
            98
          ) =>
    }
  }
}

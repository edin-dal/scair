package scair

import fastparse.*
import org.scalatest.*
import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import org.scalatest.prop.*
import org.scalatest.prop.TableDrivenPropertyChecks.forAll
import org.scalatest.prop.Tables.Table
import scair.dialects.builtin.*
import scair.ir.*

class AttrParserTest extends AnyFlatSpec with BeforeAndAfter {

  val ctx = MLContext()
  val args = scair.core.utils.Args(allow_unregistered = true)
  var parser = new Parser(ctx, args)
  var printer = new Printer(true)

  before {
    parser = new Parser(ctx, args)
    printer = new Printer(true)
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
  val I1 = IntegerType(IntData(1), Signless)
  val I16 = IntegerType(IntData(16), Signless)
  val I32 = IntegerType(IntData(32), Signless)
  val I64 = IntegerType(IntData(64), Signless)
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
    (IntegerType(IntData(874234232), Signless), "Success", "i874234232"),
    (IntegerType(IntData(7), Signed), "Success", "si7"),
    (IntegerType(IntData(8), Unsigned), "Success", "ui8"),
    (
      RankedTensorType(
        ArrayAttribute(Seq(IntData(3), IntData(-1), IntData(5))),
        Float32Type,
        None
      ),
      "Success",
      "tensor<3x?x5xf32>"
    ),
    (UnrankedTensorType(Float32Type), "Success", "tensor<*xf32>"),
    (
      ArrayAttribute(Seq(F64, ArrayAttribute(Seq()), StringData("hello"))),
      "Success",
      "[f64, [], \"hello\"]"
    ),
    (StringData("hello world!"), "Success", "\"hello world!\""),
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
    ("i874234232", "Success", IntegerType(IntData(874234232), Signless)),
    ("874234232", "Success", IntegerType(IntData(874234232), Signless)),
    ("si7", "Success", IntegerType(IntData(7), Signed)),
    ("ui8", "Success", IntegerType(IntData(8), Unsigned)),
    ("index", "Success", INDEX),
    (
      "[f64, [], \"hello\"]",
      "Success",
      ArrayAttribute(Seq(F64, ArrayAttribute(Seq()), StringData("hello")))
    ),
    ("\"hello world!\"", "Success", StringData("hello world!")),
    (
      "tensor<3x?x5xf32>",
      "Success",
      RankedTensorType(
        ArrayAttribute(Seq(IntData(3), IntData(-1), IntData(5))),
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
      input.custom_print shouldEqual expected
    }
  }

  forAll(strToAttributeTests) { (input, result, expected) =>
    // Get the expected output
    val res = getResult(result, expected)
    "strToAttributeTests" should s"[ '$input' -> '$expected' = $result ]" in {
      // Run the pqrser on the input and check
      parse(input, parser.BuiltIn(using _)) should matchPattern { case res => // pass
      }
    }
  }

  val valF32 = Value[Attribute](F32)
  val valF64 = Value[Attribute](F64)
  val valI1 = Value[Attribute](I1)
  val valI16 = Value[Attribute](I16)
  val valINDEX = Value[Attribute](INDEX)

  "printDataibutesWithinOp" should "return the correct string representation of a Operation with blocks and different attributes" in {
    val op = UnregisteredOperation(
      "test.op",
      results_types = ListType(
        F32,
        F64,
        F80
      )
    )
    val block1 = Block(
      ListType(F128),
      ListType(
        op,
        UnregisteredOperation(
          "test.op",
          operands = ListType(op.results(1), op.results(0))
        )
      )
    )
    val op2 = UnregisteredOperation(
      "test.op",
      successors = ListType(block1),
      results_types = ListType(
        I1,
        I16,
        I32
      )
    )
    val block2 = Block(
      ListType(I32),
      ListType(
        op2,
        UnregisteredOperation(
          "test.op",
          operands = ListType(
            op2.results(1),
            op2.results(0)
          )
        )
      )
    )
    val op3 = UnregisteredOperation(
      "test.op",
      results_types = ListType(
        INDEX
      )
    )
    val block3 = Block(
      ListType(I64),
      ListType(
        op3,
        UnregisteredOperation(
          "test.op",
          operands = ListType(op3.results(0))
        )
      )
    )

    val program =
      UnregisteredOperation(
        "op1",
        regions = ListType(
          Region(
            Seq(
              block1,
              block2,
              block3
            )
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

    val block1 = new Block(
      ListType(F16),
      ListType(
        UnregisteredOperation(
          "test.op",
          operands = ListType(
            valF32,
            valF64,
            Value(F80)
          )
        ),
        UnregisteredOperation(
          "test.op",
          ListType(valF64, valF32)
        )
      )
    )

    val op4 = UnregisteredOperation(
      "test.op",
      successors = ListType(block1),
      results_types = ListType(
        I1,
        I16,
        I32
      )
    )

    val block2 = new Block(
      ListType(F128),
      ListType(
        op4,
        UnregisteredOperation(
          "test.op",
          operands = ListType(
            op4.results(1),
            op4.results(0)
          )
        )
      )
    )

    val op5 = UnregisteredOperation(
      "test.op",
      results_types = ListType(
        INDEX
      )
    )

    val block3 = new Block(
      ListType(I64),
      ListType(
        op5,
        UnregisteredOperation(
          "test.op",
          operands = ListType(op5.results(0))
        )
      )
    )

    val program =
      UnregisteredOperation(
        "op1",
        regions = ListType(
          Region(
            Seq(
              block1,
              block2,
              block3
            )
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
      pattern = parser.TopLevel(using _)
    ) should matchPattern { case Parsed.Success(program, _) => }
  }

  "parsingInteger" should "match parsed string against expected string" in {

    val input = """"op1"()({
                     |  ^bb0(%0: f128):
                     |    %1, %2, %3 = "test.op"() : () -> (i32, si64, ui80)
                     |  }) : () -> ()""".stripMargin

    parser
      .parseThis(
        text = input,
        pattern = parser.OperationPat(using _)
      ) should matchPattern {
      case Parsed.Success(
            UnregisteredOperation(
              "op1",
              ListType(),
              ListType(),
              ListType(),
              ListType(
                Region(
                  Seq(
                    Block(
                      ListType(Value(F128)),
                      ListType(
                        UnregisteredOperation(
                          "test.op",
                          ListType(),
                          ListType(),
                          ListType(
                            IntegerType(IntData(32), Signless),
                            IntegerType(IntData(64), Signed),
                            IntegerType(IntData(80), Unsigned)
                          ),
                          ListType(),
                          _,
                          _
                        )
                      )
                    )
                  )
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

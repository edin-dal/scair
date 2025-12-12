package scair

import fastparse.*
import org.scalatest.*
import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import org.scalatest.prop.TableDrivenPropertyChecks.forAll
import org.scalatest.prop.Tables.Table
import scair.dialects.builtin.*
import scair.ir.*
import scair.parse.*
import java.io.*

class AttrParserTest extends AnyFlatSpec with BeforeAndAfter:

  val ctx = MLContext()
  var parser = new Parser(ctx, allowUnregisteredDialect = true)
  var out = StringWriter()
  var printer = new Printer(true, p = PrintWriter(out))

  before {
    parser = new Parser(ctx, allowUnregisteredDialect = true)
    out = StringWriter()
    printer = new Printer(true, p = PrintWriter(out))
  }

  def getResult[A](result: String, expected: A) =
    result match
      case "Success" => ((x: Int) => Parsed.Success(expected, x))
      case "Failure" => Parsed.Failure(_, _, _)

  val F16 = Float16Type()
  val F32 = Float32Type()
  val F64 = Float64Type()
  val F80 = Float80Type()
  val F128 = Float128Type()
  val I1 = IntegerType(IntData(1), Signless)
  val I16 = IntegerType(IntData(16), Signless)
  val I32 = IntegerType(IntData(32), Signless)
  val I64 = IntegerType(IntData(64), Signless)
  val INDEX = IndexType()

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
        Float32Type(),
        ArrayAttribute(Seq(IntData(3), IntData(-1), IntData(5))),
        None,
      ),
      "Success",
      "tensor<3x?x5xf32>",
    ),
    (UnrankedTensorType(Float32Type()), "Success", "tensor<*xf32>"),
    (
      ArrayAttribute(Seq(F64, ArrayAttribute(Seq()), StringData("hello"))),
      "Success",
      "[f64, [], \"hello\"]",
    ),
    (StringData("hello world!"), "Success", "\"hello world!\""),
    (INDEX, "Success", "index"),
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
      ArrayAttribute(Seq(F64, ArrayAttribute(Seq()), StringData("hello"))),
    ),
    ("\"hello world!\"", "Success", StringData("hello world!")),
    (
      "tensor<3x?x5xf32>",
      "Success",
      RankedTensorType(
        Float32Type(),
        ArrayAttribute(Seq(IntData(3), IntData(-1), IntData(5))),
        None,
      ),
    ),
    ("tensor<*xf32>", "Success", UnrankedTensorType(Float32Type())),
    ("fg12", "Failure", ""),
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
      parse(input, attributeP(using _, parser)) should matchPattern {
        case res => // pass
      }
    }
  }

  val valF32 = Value[Attribute](F32)
  val valF64 = Value[Attribute](F64)
  val valI1 = Value[Attribute](I1)
  val valI16 = Value[Attribute](I16)
  val valINDEX = Value[Attribute](INDEX)

  "printDataibutesWithinOp" should
    "return the correct string representation of a Operation with blocks and different attributes" in {
      val op = UnregisteredOperation("test.op")(
        results = Seq(
          F32,
          F64,
          F80,
        ).map(Result(_))
      )
      val block1 = Block(
        ListType(F128),
        ListType(
          op,
          UnregisteredOperation("test.op")(
            operands = Seq(op.results(1), op.results(0))
          ),
        ),
      )
      val op2 = UnregisteredOperation("test.op")(
        successors = Seq(block1),
        results = Seq(
          I1,
          I16,
          I32,
        ).map(Result(_)),
      )
      val block2 = Block(
        ListType(I32),
        ListType(
          op2,
          UnregisteredOperation("test.op")(
            operands = Seq(
              op2.results(1),
              op2.results(0),
            )
          ),
        ),
      )
      val op3 = UnregisteredOperation("test.op")(
        results = Seq(
          INDEX
        ).map(Result(_))
      )
      val block3 = Block(
        ListType(I64),
        ListType(
          op3,
          UnregisteredOperation("test.op")(
            operands = Seq(op3.results(0))
          ),
        ),
      )

      val program =
        UnregisteredOperation("op1")(
          regions = Seq(
            Region(
              Seq(
                block1,
                block2,
                block3,
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
                     |  %9 = "test.op"() : () -> index
                     |  "test.op"(%9) : (index) -> ()
                     |}) : () -> ()
                     |""".stripMargin
      printer.print(program)(using 0)
      val result = out.toString()
      result shouldEqual expected
    }

  "parseDifferentAttributes" should
    "match parsed string against expected string" in {

      val block1 = new Block(
        ListType(F16),
        ListType(
          UnregisteredOperation("test.op")(
            operands = Seq(
              valF32,
              valF64,
              Value(F80),
            )
          ),
          UnregisteredOperation("test.op")(
            Seq(valF64, valF32)
          ),
        ),
      )

      val op4 = UnregisteredOperation("test.op")(
        successors = Seq(block1),
        results = Seq(
          I1,
          I16,
          I32,
        ).map(Result(_)),
      )

      val block2 = new Block(
        ListType(F128),
        ListType(
          op4,
          UnregisteredOperation("test.op")(
            operands = Seq(
              op4.results(1),
              op4.results(0),
            )
          ),
        ),
      )

      val op5 = UnregisteredOperation("test.op")(
        results = Seq(
          INDEX
        ).map(Result(_))
      )

      val block3 = new Block(
        ListType(I64),
        ListType(
          op5,
          UnregisteredOperation("test.op")(
            operands = Seq(op5.results(0))
          ),
        ),
      )

      val program =
        UnregisteredOperation("op1")(
          regions = Seq(
            Region(
              Seq(
                block1,
                block2,
                block3,
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
                     |    %9 = "test.op"() : () -> index
                     |    "test.op"(%9) : (index) -> ()
                     |  }) : () -> ()""".stripMargin

      parser.parse(
        input = input
      ) should matchPattern { case Parsed.Success(program, _) => }
    }

  "parsingInteger" should "match parsed string against expected string" in {

    val input = """"op1"()({
                     |  ^bb0(%0: f128):
                     |    %1, %2, %3 = "test.op"() : () -> (i32, si64, ui80)
                     |  }) : () -> ()""".stripMargin

    parser.parse(
      input = input,
      parser = operationP(using _, parser),
    ) should matchPattern {
      case Parsed.Success(
            UnregisteredOperation(
              "op1",
              Seq(),
              Seq(),
              Seq(),
              Seq(
                Region(
                  Block(
                    ListType(Value(F128)),
                    BlockOperations(
                      UnregisteredOperation(
                        "test.op",
                        Seq(),
                        Seq(),
                        Seq(
                          Result(IntegerType(IntData(32), Signless)),
                          Result(IntegerType(IntData(64), Signed)),
                          Result(IntegerType(IntData(80), Unsigned)),
                        ),
                        Seq(),
                        _,
                        _,
                      )
                    ),
                  )
                )
              ),
              _,
              _,
            ),
            98,
          ) =>
    }
  }

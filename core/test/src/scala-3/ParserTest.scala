package scair

import fastparse.*
import org.scalatest.*
import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import org.scalatest.prop.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.parse.*

class ParserTest
    extends AnyFlatSpec
    with BeforeAndAfter
    with TableDrivenPropertyChecks:

  val I32 = IntegerType(IntData(32), Signless)
  val I64 = IntegerType(IntData(64), Signless)

  def getResult[A](result: String, expected: A) =
    result match
      case "Success" => ((x: Int) => Parsed.Success(expected, x))
      case "Failure" => Parsed.Failure(_, _, _)

  def passed[A](res: Parsed[A], expectedResult: A, result: String) =
    res match
      case Parsed.Success(actualResult, i) if result == "Success" =>
        expectedResult == actualResult
      case Parsed.Failure(_, _, _) if result == "Failure" => true
      case _                                              => false

  val ctx = new MLContext()
  var parser: Parser = new Parser(ctx, allowUnregisteredDialect = true)

  before {
    parser = new Parser(ctx, allowUnregisteredDialect = true)
  }

  val digitTests = Table(
    ("input", "result", "expected"),
    ("7", "Success", ()),
    ("a", "Failure", ""),
    (" $ ! £ 4 1 ", "Failure", ""),
  )

  val hexTests = Table(
    ("input", "result", "expected"),
    ("5", "Success", ()),
    ("f", "Success", ()),
    ("E", "Success", ()),
    ("41", "Success", ()),
    ("G", "Failure", ""),
    ("g", "Failure", ""),
  )

  val intLiteralTests = Table(
    ("input", "result", "expected"),
    ("123456789", "Success", 123456789),
    ("1231f", "Success", 1231),
    ("0x0011ffff", "Success", 0x0011ffff),
    ("1xds%", "Success", 1),
    ("0xgg", "Success", 0),
    ("f1231", "Failure", ""),
    ("0x0011gggg", "Success", 0x0011),
  )

  val decimalLiteralTests = Table(
    ("input", "result", "expected"),
    ("123456789", "Success", 123456789),
    ("1231f", "Success", 1231),
    ("f1231", "Failure", ""),
  )

  val hexadecimalLiteralTests = Table(
    ("input", "result", "expected"),
    ("0x0011ffff", "Success", 0x0011ffff),
    ("0x0011gggg", "Success", 0x0011),
    ("1xds%", "Failure", ""),
    ("0xgg", "Failure", ""),
  )

  val floatLiteralTests = Table(
    ("input", "result", "expected"),
    ("1.0", "Success", 1.0),
    ("1.01242", "Success", 1.01242),
    ("993.013131", "Success", 993.013131),
    ("1.0e10", "Success", 1.0e10),
    ("1.0E10", "Success", 1.0e10),
    ("1.", "Failure", ""),
  )

  val stringLiteralTests = Table(
    ("input", "result", "expected"),
    ("\"hello\"", "Success", "hello"),
  )

  val valueIdTests = Table(
    ("input", "result", "expected"),
    ("%hello", "Success", "hello"),
    ("%Ater", "Success", "Ater"),
    ("%312321", "Success", "312321"),
    ("%$$$$$", "Success", "$$$$$"),
    ("%_-_-_", "Success", "_-_-_"),
    ("%3asada", "Success", "3"),
    ("% hello", "Failure", ""),
  )

  val unitTests = Table(
    ("name", "pattern", "tests"),
    (
      "Digit",
      ((x: fastparse.P[?]) => decDigitsP(using x)),
      digitTests,
    ),
    (
      "HexDigit",
      ((x: fastparse.P[?]) => hexDigitsP(using x)),
      hexTests,
    ),
    (
      "IntegerLiteral",
      ((x: fastparse.P[?]) => integerLiteralP(using x)),
      intLiteralTests,
    ),
    (
      "DecimalLiteral",
      ((x: fastparse.P[?]) => decimalLiteralP(using x)),
      decimalLiteralTests,
    ),
    (
      "HexadecimalLiteral",
      ((x: fastparse.P[?]) => hexadecimalLiteralP(using x)),
      hexadecimalLiteralTests,
    ),
    (
      "FloatLiteral",
      ((x: fastparse.P[?]) => floatLiteralP(using x)),
      floatLiteralTests,
    ),
    (
      "StringLiteral",
      ((x: fastparse.P[?]) => stringLiteralP(using x)),
      stringLiteralTests,
    ),
    (
      "ValueId",
      ((x: fastparse.P[?]) => valueIdP(using x)),
      valueIdTests,
    ),
  )

  forAll(unitTests) { (name, pattern, tests) =>
    forAll(tests) { (input, result, expected) =>
      // Get the expected output
      val res = getResult(result, expected)
      name should s"[ '$input' -> '$expected' = $result ]" in {
        // Run the pqrser on the input and check
        parser.parse(input, pattern) should matchPattern {
          case x if passed(x.asInstanceOf[Parsed[?]], expected, result) =>
        }
      }
    }
  }

  "Block - Unit Tests" should "parse correctly" in withClue("Test 1: ") {
    parser.parse(
      input = "^bb0(%5: i32):\n" +
        "%0, %1, %2 = \"test.op\"() : () -> (i32, i64, i32)\n" +
        "\"test.op\"(%1, %0) : (i64, i32) -> ()",
      parser = blockP(using _, parser),
    ) should matchPattern {
      case Parsed.Success(
            Block(
              ListType(Value(I32)),
              BlockOperations(
                UnregisteredOperation(
                  "test.op",
                  Seq(),
                  Seq(),
                  Seq(
                    Result(I32),
                    Result(I64),
                    Result(I32),
                  ),
                  Seq(),
                  _,
                  _,
                ),
                UnregisteredOperation(
                  "test.op",
                  Seq(Value(I64), Value(I32)),
                  Seq(),
                  Seq(),
                  Seq(),
                  _,
                  _,
                ),
              ),
            ),
            100,
          ) =>
    }
  }

  "Region - Unit Tests" should "parse correctly" in withClue("Test 1: ") {
    parser.parse(
      input = "{^bb0(%5: i32):\n" +
        "%0, %1, %2 = \"test.op\"() : () -> (i32, i64, i32)\n" +
        "\"test.op\"(%1, %0) : (i64, i32) -> ()" + "^bb1(%4: i32):\n" +
        "%7, %8, %9 = \"test.op\"() : () -> (i32, i64, i32)\n" +
        "\"test.op\"(%8, %7) : (i64, i32) -> ()" + "}",
      parser = regionP()(using _, parser),
    ) should matchPattern {
      case Parsed.Success(
            Region(
              Block(
                ListType(Value(I32)),
                BlockOperations(
                  UnregisteredOperation(
                    "test.op",
                    Seq(),
                    Seq(),
                    Seq(
                      Result(I32),
                      Result(I64),
                      Result(I32),
                    ),
                    Seq(),
                    _,
                    _,
                  ),
                  UnregisteredOperation(
                    "test.op",
                    Seq(Value(I64), Value(I32)),
                    Seq(),
                    Seq(),
                    Seq(),
                    _,
                    _,
                  ),
                ),
              ),
              Block(
                ListType(Value(I32)),
                BlockOperations(
                  UnregisteredOperation(
                    "test.op",
                    Seq(),
                    Seq(),
                    Seq(
                      Result(I32),
                      Result(I64),
                      Result(I32),
                    ),
                    Seq(),
                    _,
                    _,
                  ),
                  UnregisteredOperation(
                    "test.op",
                    Seq(
                      Value(I64),
                      Value(I32),
                    ),
                    Seq(),
                    Seq(),
                    Seq(),
                    _,
                    _,
                  ),
                ),
              ),
            ),
            202,
          ) =>
    }
  }

  "Region2 - Unit Tests" should "parse correctly" in withClue("Test 2: ") {
    parser.parse(
      input = """{
^bb0(%5: i32):
  %0, %1, %2 = "test.op"() : () -> (i32, i64, i32)
  "test.op"(%1, %0) : (i64, i32) -> ()
^bb0(%4: i32):
  %7, %8, %9 = "test.op"() : () -> (i32, i64, i32)
  "test.op"(%8, %7) : (i64, i32) -> ()
}""",
      parser = regionP()(using _, parser),
      verboseFailures = true,
    ) should matchPattern {
      case Parsed.Failure(
            "Block cannot be defined twice within the same scope - ^bb0",
            _,
            _,
          ) =>
    }
  }

  "Operation  Test - Failure" should "Test faulty Operation IR" in
    withClue("Test 2: ") {

      val text = """"op1"()[^bb3]({
                   |^bb3(%4: i32):
                   |  %5, %6, %7 = "test.op"() : () -> (i32, i64, i32)
                   |  "test.op"(%6, %5) : (i64, i32) -> ()
                   |^bb4(%8: i32):
                   |  %9, %10, %11 = "test.op"() : () -> (i32, i64, i32)
                   |  "test.op"(%10, %9) : (i64, i32) -> ()
                   |}) : () -> ()""".stripMargin

      parser.parse(
        input = text,
        parser = moduleP(using _, parser),
        true,
      ) should matchPattern {
        case Parsed.Failure("Successor ^bb3 not defined within Scope", _, _) =>
      }
    }

  "Operation  Test" should "Test forward block reference" in
    withClue("Test 3:") {
      val text = """"op1"()({
                 |  ^bb3():
                 |    "test.op"()[^bb4] : () -> ()
                 |  ^bb4():
                 |    "test.op"() : () -> ()
                 |  }) : () -> ()""".stripMargin

      val bb4 = Block(
        ListType(UnregisteredOperation("test.op")())
      )
      val bb3 = Block(
        ListType(UnregisteredOperation("test.op")(successors = Seq(bb4)))
      )
      val operation =
        UnregisteredOperation("test.op")(
          regions = Seq(Region(bb3, bb4))
        )

      parser.parse(
        input = text,
        parser = operationP(using _, parser),
      ) should matchPattern { case operation =>
      }
    }

  "TopLevel Tests" should "Test full programs" in withClue("Test 1: ") {
    parser.parse(
      input = "%0, %1, %2 = \"test.op\"() : () -> (i32, i64, i32)",
      parser = moduleP(using _, parser),
    ) should matchPattern {
      case Parsed.Success(
            ModuleOp(
              Region(
                Block(
                  ListType(),
                  BlockOperations(
                    UnregisteredOperation(
                      "test.op",
                      Seq(),
                      Seq(),
                      Seq(
                        Result(I32),
                        Result(I64),
                        Result(I32),
                      ),
                      Seq(),
                      _,
                      _,
                    )
                  ),
                )
              )
            ),
            48,
          ) =>
    }
  }

  "Value Uses assignment test forward ref" should
    "Test Operation's  forward-referenced Operand uses" in
    withClue("Operand Uses: ") {

      val text = """  "op1"(%0, %1, %2) : (i32, i64, i32) -> ()
                    | "op2"(%0, %1, %2) : (i32, i64, i32) -> ()
                    | "op3"(%0, %1, %2) : (i32, i64, i32) -> ()
                    | "op4"(%0, %1, %2) : (i32, i64, i32) -> ()
                    | %0, %1, %2 = "test.op"() : () -> (i32, i64, i32)"""
        .stripMargin

      val Parsed.Success(value, _) = parser.parse(
        input = text,
        parser = moduleP(using _, parser),
      ): @unchecked

      val uses0 = value.regions(0).blocks(0).operations(4).results(0).uses
      val uses1 = value.regions(0).blocks(0).operations(4).results(1).uses
      val uses2 = value.regions(0).blocks(0).operations(4).results(2).uses

      uses0.size shouldEqual 4
      uses0.map(use => (use.operation.name, use.index)) shouldEqual
        Set(("op1", 0), ("op2", 0), ("op3", 0), ("op4", 0))

      uses1.size shouldEqual 4
      uses1.map(use => (use.operation.name, use.index)) shouldEqual
        Set(("op1", 1), ("op2", 1), ("op3", 1), ("op4", 1))

      uses2.size shouldEqual 4
      uses2.map(use => (use.operation.name, use.index)) shouldEqual
        Set(("op1", 2), ("op2", 2), ("op3", 2), ("op4", 2))
    }

  "Value Uses assignment test" should "Test Operation's Operand uses" in
    withClue("Operand Uses: ") {

      val text = """  %0, %1, %2 = "test.op"() : () -> (i32, i64, i32)
                    | "op1"(%0, %1, %2) : (i32, i64, i32) -> ()
                    | "op2"(%0, %1, %2) : (i32, i64, i32) -> ()
                    | "op3"(%0, %1, %2) : (i32, i64, i32) -> ()
                    | "op4"(%0, %1, %2) : (i32, i64, i32) -> ()""".stripMargin

      val Parsed.Success(value, _) = parser.parse(
        input = text,
        parser = moduleP(using _, parser),
      ): @unchecked

      val uses0 = value.regions(0).blocks(0).operations(0).results(0).uses
      val uses1 = value.regions(0).blocks(0).operations(0).results(1).uses
      val uses2 = value.regions(0).blocks(0).operations(0).results(2).uses

      uses0.size shouldEqual 4
      uses0.map(use => (use.operation.name, use.index)) shouldEqual
        Set(("op1", 0), ("op2", 0), ("op3", 0), ("op4", 0))

      uses1.size shouldEqual 4
      uses1.map(use => (use.operation.name, use.index)) shouldEqual
        Set(("op1", 1), ("op2", 1), ("op3", 1), ("op4", 1))

      uses2.size shouldEqual 4
      uses2.map(use => (use.operation.name, use.index)) shouldEqual
        Set(("op1", 2), ("op2", 2), ("op3", 2), ("op4", 2))
    }

  "Operation Erasure" should "Test that operation gets erased :)" in
    withClue("Operand Erasure: ") {

      val text = """  %0, %1 = "test.op"() : () -> (i32, i64)
                    | %2 = "test.op"(%0) : (i32) -> i32
                    | "op1"(%0, %1, %2) : (i32, i64, i32) -> ()
                    | "op2"(%0, %1, %2) : (i32, i64, i32) -> ()
                    | "op3"(%0, %1, %2) : (i32, i64, i32) -> ()
                    | "op4"(%0, %1, %2) : (i32, i64, i32) -> ()""".stripMargin

      val Parsed.Success(value, _) = parser.parse(
        input = text,
        parser = moduleP(using _, parser),
      ): @unchecked

      val printer = new Printer(true)

      val opToErase = value.regions(0).blocks(0).operations(1)

      val block =
        opToErase.containerBlock.getOrElse(throw new Exception("bruh"))

      val exception = intercept[Exception](
        block.eraseOp(opToErase)
      ).getMessage shouldBe
        "Attempting to erase a Value that has uses in other operations."

      opToErase.containerBlock shouldEqual None
    }

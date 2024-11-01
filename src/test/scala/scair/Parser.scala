package scair

import org.scalatest._
import Inspectors._
import flatspec._
import matchers.should.Matchers._
import prop._
import Tables._
import exceptions._

import fastparse._, MultiLineWhitespace._
import scala.collection.mutable
import scala.util.{Try, Success, Failure}
import Parser._
import AttrParser._

import scair.ir._
import scair.dialects.builtin._

class ParserTest
    extends AnyFlatSpec
    with BeforeAndAfter
    with TableDrivenPropertyChecks {

  val I32 = IntegerType(IntData(32), Signless)
  val I64 = IntegerType(IntData(64), Signless)

  def getResult[A](result: String, expected: A) =
    result match {
      case "Success" => ((x: Int) => Parsed.Success(expected, x))
      case "Failure" => Parsed.Failure(_, _, _)
    }

  var parser: Parser = new Parser

  before {
    parser = new Parser
  }

  val digitTests = Table(
    ("input", "result", "expected"),
    ("7", "Success", "7"),
    ("a", "Failure", ""),
    (" $ ! £ 4 1 ", "Failure", "")
  )

  val hexTests = Table(
    ("input", "result", "expected"),
    ("5", "Success", "5"),
    ("f", "Success", "f"),
    ("E", "Success", "E"),
    ("41", "Success", "4"),
    ("G", "Failure", ""),
    ("g", "Failure", "")
  )

  val letterTests = Table(
    ("input", "result", "expected"),
    ("a", "Success", "a"),
    ("G", "Success", "G"),
    ("4", "Failure", "")
  )

  val idPunctTests = Table(
    ("input", "result", "expected"),
    ("$", "Success", "$"),
    (".", "Success", "."),
    ("_", "Success", "_"),
    ("-", "Success", "-"),
    ("%", "Failure", ""),
    ("£", "Failure", ""),
    ("dfd", "Failure", ""),
    ("0", "Failure", "")
  )

  val intLiteralTests = Table(
    ("input", "result", "expected"),
    ("123456789", "Success", "123456789"),
    ("1231f", "Success", "1231"),
    ("0x0011ffff", "Success", "0x0011ffff"),
    ("1xds%", "Success", "1"),
    ("0xgg", "Success", "0"),
    ("f1231", "Failure", ""),
    ("0x0011gggg", "Failure", "")
  )

  val decimalLiteralTests = Table(
    ("input", "result", "expected"),
    ("123456789", "Success", "123456789"),
    ("1231f", "Success", "1231"),
    ("f1231", "Failure", "")
  )

  val hexadecimalLiteralTests = Table(
    ("input", "result", "expected"),
    ("0x0011ffff", "Success", "0x0011ffff"),
    ("0x0011gggg", "Failure", ""),
    ("1xds%", "Failure", ""),
    ("0xgg", "Failure", "")
  )

  val floatLiteralTests = Table(
    ("input", "result", "expected"),
    ("1.0", "Success", "1.0"),
    ("1.01242", "Success", "1.01242"),
    ("993.013131", "Success", "993.013131"),
    ("1.0e10", "Success", "1.0e10"),
    ("1.0E10", "Success", "1.0E10"),
    ("1.", "Failure", "")
  )

  val stringLiteralTests = Table(
    ("input", "result", "expected"),
    ("\"hello\"", "Success", "hello")
  )

  val valueIdTests = Table(
    ("input", "result", "expected"),
    ("%hello", "Success", "hello"),
    ("%Ater", "Success", "Ater"),
    ("%312321", "Success", "312321"),
    ("%$$$$$", "Success", "$$$$$"),
    ("%_-_-_", "Success", "_-_-_"),
    ("%3asada", "Success", "3"),
    ("% hello", "Failure", "")
  )

  val opResultListTests = Table(
    ("input", "result", "expected"),
    ("%0, %1, %2 =", "Success", List("0", "1", "2")),
    ("%0   ,    %1   ,   %2  =   ", "Success", List("0", "1", "2")),
    ("%0,%1,%2=", "Success", List("0", "1", "2"))
  )

  val unitTests = Table(
    ("name", "pattern", "tests"),
    (
      "Digit",
      ((x: fastparse.P[_]) => Parser.Digit(x)),
      digitTests
    ),
    (
      "HexDigit",
      ((x: fastparse.P[_]) => Parser.HexDigit(x)),
      hexTests
    ),
    (
      "Letter",
      ((x: fastparse.P[_]) => Parser.Letter(x)),
      letterTests
    ),
    (
      "IdPunct",
      ((x: fastparse.P[_]) => Parser.IdPunct(x)),
      idPunctTests
    ),
    (
      "IntegerLiteral",
      ((x: fastparse.P[_]) => Parser.IntegerLiteral(x)),
      intLiteralTests
    ),
    (
      "DecimalLiteral",
      ((x: fastparse.P[_]) => Parser.DecimalLiteral(x)),
      decimalLiteralTests
    ),
    (
      "HexadecimalLiteral",
      ((x: fastparse.P[_]) => Parser.HexadecimalLiteral(x)),
      hexadecimalLiteralTests
    ),
    (
      "FloatLiteral",
      ((x: fastparse.P[_]) => Parser.FloatLiteral(x)),
      floatLiteralTests
    ),
    (
      "StringLiteral",
      ((x: fastparse.P[_]) => Parser.StringLiteral(x)),
      stringLiteralTests
    ),
    (
      "ValueId",
      ((x: fastparse.P[_]) => Parser.ValueId(x)),
      valueIdTests
    ),
    (
      "OpResultList",
      ((x: fastparse.P[_]) => Parser.OpResultList(x)),
      opResultListTests
    )
  )

  forAll(unitTests) { (name, pattern, tests) =>
    forAll(tests) { (input, result, expected) =>
      // Get the expected output
      val res = getResult(result, expected)
      name should s"[ '$input' -> '$expected' = $result ]" in {
        // Run the pqrser on the input and check
        parser.parseThis(input, pattern) should matchPattern {
          case res => // pass
        }
      }
    }
  }

  "Block - Unit Tests" should "parse correctly" in {
    withClue("Test 1: ") {
      parser.parseThis(
        text =
          "^bb0(%5: i32):\n" + "%0, %1, %2 = \"test.op\"() : () -> (i32, i64, i32)\n" +
            "\"test.op\"(%1, %0) : (i64, i32) -> ()",
        pattern = parser.Block(_)
      ) should matchPattern {
        case Parsed.Success(
              Block(
                ListType(
                  UnregisteredOperation(
                    "test.op",
                    ListType(),
                    ListType(),
                    ListType(
                      Value(I32),
                      Value(I64),
                      Value(I32)
                    ),
                    ListType(),
                    _,
                    _
                  ),
                  UnregisteredOperation(
                    "test.op",
                    ListType(Value(I64), Value(I32)),
                    ListType(),
                    ListType(),
                    ListType(),
                    _,
                    _
                  )
                ),
                ListType(Value(I32))
              ),
              100
            ) =>
      }
    }
  }

  "Region - Unit Tests" should "parse correctly" in {

    withClue("Test 1: ") {
      parser.parseThis(
        text =
          "{^bb0(%5: i32):\n" + "%0, %1, %2 = \"test.op\"() : () -> (i32, i64, i32)\n" +
            "\"test.op\"(%1, %0) : (i64, i32) -> ()" + "^bb1(%4: i32):\n" + "%7, %8, %9 = \"test.op\"() : () -> (i32, i64, i32)\n" +
            "\"test.op\"(%8, %7) : (i64, i32) -> ()" + "}",
        pattern = parser.Region(_)
      ) should matchPattern {
        case Parsed.Success(
              Region(
                Seq(
                  Block(
                    ListType(
                      UnregisteredOperation(
                        "test.op",
                        ListType(),
                        ListType(),
                        ListType(
                          Value(I32),
                          Value(I64),
                          Value(I32)
                        ),
                        ListType(),
                        _,
                        _
                      ),
                      UnregisteredOperation(
                        "test.op",
                        ListType(Value(I64), Value(I32)),
                        ListType(),
                        ListType(),
                        ListType(),
                        _,
                        _
                      )
                    ),
                    ListType(Value(I32))
                  ),
                  Block(
                    ListType(
                      UnregisteredOperation(
                        "test.op",
                        ListType(),
                        ListType(),
                        ListType(
                          Value(I32),
                          Value(I64),
                          Value(I32)
                        ),
                        ListType(),
                        _,
                        _
                      ),
                      UnregisteredOperation(
                        "test.op",
                        ListType(
                          Value(I64),
                          Value(I32)
                        ),
                        ListType(),
                        ListType(),
                        ListType(),
                        _,
                        _
                      )
                    ),
                    ListType(Value(I32))
                  )
                )
              ),
              202
            ) =>
      }
    }
  }

  "Region2 - Unit Tests" should "parse correctly" in {

    withClue("Test 2: ") {
      val exception = intercept[Exception](
        parser.parseThis(
          text =
            "{^bb0(%5: i32):\n" + "%0, %1, %2 = \"test.op\"() : () -> (i32, i64, i32)\n" +
              "\"test.op\"(%1, %0) : (i64, i32) -> ()" + "^bb0(%4: i32):\n" + "%7, %8, %9 = \"test.op\"() : () -> (i32, i64, i32)\n" +
              "\"test.op\"(%8, %7) : (i64, i32) -> ()" + "}",
          pattern = parser.Region(_)
        )
      ).getMessage shouldBe "Block cannot be defined twice within the same scope - ^bb0"
    }
  }

  "Operation  Test - Failure" should "Test faulty Operation IR" in {
    withClue("Test 2: ") {

      val text = """"op1"()[^bb3]({
                   |^bb3(%4: i32):
                   |  %5, %6, %7 = "test.op"() : () -> (i32, i64, i32)
                   |  "test.op"(%6, %5) : (i64, i32) -> ()
                   |^bb4(%8: i32):
                   |  %9, %10, %11 = "test.op"() : () -> (i32, i64, i32)
                   |  "test.op"(%10, %9) : (i64, i32) -> ()
                   |}) : () -> ()""".stripMargin

      val exception = intercept[Exception](
        parser.parseThis(
          text = text,
          pattern = parser.TopLevel(_)
        )
      ).getMessage shouldBe "Successor ^bb3 not defined within Scope"
    }
  }

  "Operation  Test" should "Test forward block reference" in {
    withClue("Test 3:") {
      val text = """"op1"()({
                 |  ^bb3():
                 |    "test.op"()[^bb4] : () -> ()
                 |  ^bb4():
                 |    "test.op"() : () -> ()
                 |  }) : () -> ()""".stripMargin

      val bb4 = Block(
        ListType(UnregisteredOperation("test.op"))
      )
      val bb3 = Block(
        ListType(UnregisteredOperation("test.op", successors = ListType(bb4)))
      )
      val operation =
        UnregisteredOperation(
          "test.op",
          regions = ListType(Region(Seq(bb3, bb4)))
        )

      parser.parseThis(
        text = text,
        pattern = parser.OperationPat(_)
      ) should matchPattern { case operation =>
      }
    }
  }

  "TopLevel Tests" should "Test full programs" in {
    withClue("Test 1: ") {
      parser.parseThis(
        text = "%0, %1, %2 = \"test.op\"() : () -> (i32, i64, i32)",
        pattern = parser.TopLevel(_)
      ) should matchPattern {
        case Parsed.Success(
              ModuleOp(
                ListType(),
                ListType(),
                ListType(),
                ListType(
                  Region(
                    Seq(
                      Block(
                        ListType(
                          UnregisteredOperation(
                            "test.op",
                            ListType(),
                            ListType(),
                            ListType(
                              Value(I32),
                              Value(I64),
                              Value(I32)
                            ),
                            ListType(),
                            _,
                            _
                          )
                        ),
                        ListType()
                      )
                    )
                  )
                ),
                _,
                _
              ),
              48
            ) =>
      }
    }
  }

  "Value Uses assignment test forward ref" should "Test Operation's Operand uses" in {
    withClue("Operand Uses: ") {

      val text = """  "op1"(%0, %1, %2) : (i32, i64, i32) -> ()
                    | "op2"(%0, %1, %2) : (i32, i64, i32) -> ()
                    | "op3"(%0, %1, %2) : (i32, i64, i32) -> ()
                    | "op4"(%0, %1, %2) : (i32, i64, i32) -> ()
                    | %0, %1, %2 = "test.op"() : () -> (i32, i64, i32)""".stripMargin

      val Parsed.Success(value, _) = parser.parseThis(
        text = text,
        pattern = parser.TopLevel(_)
      )

      val uses0 = value.regions(0).blocks(0).operations(4).results(0).uses
      val uses1 = value.regions(0).blocks(0).operations(4).results(1).uses
      val uses2 = value.regions(0).blocks(0).operations(4).results(2).uses

      uses0.length shouldEqual 4
      uses0(0).operation.name shouldEqual "op1"
      uses0(0).index shouldEqual 0
      uses0(1).operation.name shouldEqual "op2"
      uses0(1).index shouldEqual 0
      uses0(2).operation.name shouldEqual "op3"
      uses0(2).index shouldEqual 0
      uses0(3).operation.name shouldEqual "op4"
      uses0(3).index shouldEqual 0

      uses1.length shouldEqual 4
      uses1(0).operation.name shouldEqual "op1"
      uses1(0).index shouldEqual 1
      uses1(1).operation.name shouldEqual "op2"
      uses1(1).index shouldEqual 1
      uses1(2).operation.name shouldEqual "op3"
      uses1(2).index shouldEqual 1
      uses1(3).operation.name shouldEqual "op4"
      uses1(3).index shouldEqual 1

      uses2.length shouldEqual 4
      uses2(0).operation.name shouldEqual "op1"
      uses2(0).index shouldEqual 2
      uses2(1).operation.name shouldEqual "op2"
      uses2(1).index shouldEqual 2
      uses2(2).operation.name shouldEqual "op3"
      uses2(2).index shouldEqual 2
      uses2(3).operation.name shouldEqual "op4"
      uses2(3).index shouldEqual 2

    }
  }

  "Value Uses assignment test" should "Test Operation's Operand uses" in {
    withClue("Operand Uses: ") {

      val text = """  %0, %1, %2 = "test.op"() : () -> (i32, i64, i32)
                    | "op1"(%0, %1, %2) : (i32, i64, i32) -> ()
                    | "op2"(%0, %1, %2) : (i32, i64, i32) -> ()
                    | "op3"(%0, %1, %2) : (i32, i64, i32) -> ()
                    | "op4"(%0, %1, %2) : (i32, i64, i32) -> ()""".stripMargin

      val Parsed.Success(value, _) = parser.parseThis(
        text = text,
        pattern = parser.TopLevel(_)
      )

      val uses0 = value.regions(0).blocks(0).operations(0).results(0).uses
      val uses1 = value.regions(0).blocks(0).operations(0).results(1).uses
      val uses2 = value.regions(0).blocks(0).operations(0).results(2).uses

      uses0.length shouldEqual 4
      uses0(0).operation.name shouldEqual "op1"
      uses0(0).index shouldEqual 0
      uses0(1).operation.name shouldEqual "op2"
      uses0(1).index shouldEqual 0
      uses0(2).operation.name shouldEqual "op3"
      uses0(2).index shouldEqual 0
      uses0(3).operation.name shouldEqual "op4"
      uses0(3).index shouldEqual 0

      uses1.length shouldEqual 4
      uses1(0).operation.name shouldEqual "op1"
      uses1(0).index shouldEqual 1
      uses1(1).operation.name shouldEqual "op2"
      uses1(1).index shouldEqual 1
      uses1(2).operation.name shouldEqual "op3"
      uses1(2).index shouldEqual 1
      uses1(3).operation.name shouldEqual "op4"
      uses1(3).index shouldEqual 1

      uses2.length shouldEqual 4
      uses2(0).operation.name shouldEqual "op1"
      uses2(0).index shouldEqual 2
      uses2(1).operation.name shouldEqual "op2"
      uses2(1).index shouldEqual 2
      uses2(2).operation.name shouldEqual "op3"
      uses2(2).index shouldEqual 2
      uses2(3).operation.name shouldEqual "op4"
      uses2(3).index shouldEqual 2

    }
  }

  "Operation Erasure" should "Test that operation gets erased :)" in {
    withClue("Operand Erasure: ") {

      val text = """  %0, %1 = "test.op"() : () -> (i32, i64)
                    | %2 = "test.op"(%0) : (i32) -> (i32)
                    | "op1"(%0, %1, %2) : (i32, i64, i32) -> ()
                    | "op2"(%0, %1, %2) : (i32, i64, i32) -> ()
                    | "op3"(%0, %1, %2) : (i32, i64, i32) -> ()
                    | "op4"(%0, %1, %2) : (i32, i64, i32) -> ()""".stripMargin

      val Parsed.Success(value, _) = parser.parseThis(
        text = text,
        pattern = parser.TopLevel(_)
      )

      val printer = new Printer(true)

      val opToErase = value.regions(0).blocks(0).operations(1)

      val block =
        opToErase.container_block.getOrElse(throw new Exception("bruh"))

      val exception = intercept[Exception](
        block.erase_op(opToErase)
      ).getMessage shouldBe "Attempting to erase a Value that has uses in other operations."

      opToErase.container_block shouldEqual None
    }
  }
}

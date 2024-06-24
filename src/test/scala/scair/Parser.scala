package scair

import fastparse._, MultiLineWhitespace._
import scala.collection.mutable
import scala.util.{Try, Success, Failure}
import org.scalatest._
import Matchers._
import Parser._
import org.scalatest.prop.Tables.Table
import org.scalatest.prop.TableDrivenPropertyChecks.forAll
import AttrParser._

class ParserTest extends FlatSpec with BeforeAndAfter {

  val I32 = IntegerType(32, Signless)
  val I64 = IntegerType(64, Signless)

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
      ((x: fastparse.P[_]) => parser.Digit(x)),
      digitTests
    ),
    (
      "HexDigit",
      ((x: fastparse.P[_]) => parser.HexDigit(x)),
      hexTests
    ),
    (
      "Letter",
      ((x: fastparse.P[_]) => parser.Letter(x)),
      letterTests
    ),
    (
      "IdPunct",
      ((x: fastparse.P[_]) => parser.IdPunct(x)),
      idPunctTests
    ),
    (
      "IntegerLiteral",
      ((x: fastparse.P[_]) => parser.IntegerLiteral(x)),
      intLiteralTests
    ),
    (
      "DecimalLiteral",
      ((x: fastparse.P[_]) => parser.DecimalLiteral(x)),
      decimalLiteralTests
    ),
    (
      "HexadecimalLiteral",
      ((x: fastparse.P[_]) => parser.HexadecimalLiteral(x)),
      hexadecimalLiteralTests
    ),
    (
      "FloatLiteral",
      ((x: fastparse.P[_]) => parser.FloatLiteral(x)),
      floatLiteralTests
    ),
    (
      "StringLiteral",
      ((x: fastparse.P[_]) => parser.StringLiteral(x)),
      stringLiteralTests
    ),
    (
      "ValueId",
      ((x: fastparse.P[_]) => parser.ValueId(x)),
      valueIdTests
    ),
    (
      "OpResultList",
      ((x: fastparse.P[_]) => parser.OpResultList(x)),
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
                Seq(
                  Operation(
                    "test.op",
                    Seq(),
                    Seq(),
                    Seq(
                      Value(I32),
                      Value(I64),
                      Value(I32)
                    ),
                    Seq()
                  ),
                  Operation(
                    "test.op",
                    Seq(Value(I64), Value(I32)),
                    Seq(),
                    Seq(),
                    Seq()
                  )
                ),
                Seq(Value(I32))
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
                    Seq(
                      Operation(
                        "test.op",
                        Seq(),
                        Seq(),
                        Seq(
                          Value(I32),
                          Value(I64),
                          Value(I32)
                        ),
                        Seq()
                      ),
                      Operation(
                        "test.op",
                        Seq(Value(I64), Value(I32)),
                        Seq(),
                        Seq(),
                        Seq()
                      )
                    ),
                    Seq(Value(I32))
                  ),
                  Block(
                    Seq(
                      Operation(
                        "test.op",
                        Seq(),
                        Seq(),
                        Seq(
                          Value(I32),
                          Value(I64),
                          Value(I32)
                        ),
                        Seq()
                      ),
                      Operation(
                        "test.op",
                        Seq(
                          Value(I64),
                          Value(I32)
                        ),
                        Seq(),
                        Seq(),
                        Seq()
                      )
                    ),
                    Seq(Value(I32))
                  )
                ),
                None
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

  "Operation  Test" should "Test operations" in {
    withClue("Test 1: ") {
      val text = """"op1"()({
                       |  ^bb3(%4: i32):
                       |    %5, %6, %7 = "test.op"() : () -> (i32, i64, i32)
                       |    "test.op"(%6, %5) : (i64, i32) -> ()
                       |  ^bb4(%8: i32):
                       |    %9, %10, %11 = "test.op"()[^bb3] : () -> (i32, i64, i32)
                       |    "test.op"(%10, %9) : (i64, i32) -> ()
                       |  }) : () -> ()""".stripMargin

      val val1 = Value(I32)
      val val2 = Value(I64)
      val val3 = Value(I32)
      val val4 = Value(I64)

      val successorTestBlock = Block(
        Seq(
          Operation(
            "test.op",
            Seq(),
            Seq(),
            Seq(
              val3,
              val4,
              Value(I32)
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
        Seq(Value(I32))
      )

      val operation =
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
                        Value(I32)
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
                  Seq(Value(I32))
                )
              ),
              None
            )
          )
        )

      parser.parseThis(
        text = text,
        pattern = parser.OperationPat(_)
      ) should matchPattern { case operation =>
      }
    }
  }

  "Operation  Test - Failure" should "Test faulty Operation IR" in {
    withClue("Test 2: ") {

      val text = """"op1"()[^bb3]({
                   |  ^bb3(%4: i32):
                   |    %5, %6, %7 = "test.op"() : () -> (i32, i64, i32)
                   |    "test.op"(%6, %5) : (i64, i32) -> ()
                   |  ^bb4(%8: i32):
                   |    %9, %10, %11 = "test.op"() : () -> (i32, i64, i32)
                   |    "test.op"(%10, %9) : (i64, i32) -> ()
                   |  }) : () -> ()""".stripMargin

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
        Seq(Operation("test.op", Seq(), Seq(), Seq(), Seq())),
        Seq()
      )
      val bb3 = Block(
        Seq(Operation("test.op", Seq(), Seq(bb4), Seq(), Seq())),
        Seq()
      )
      val operation =
        Operation("test.op", Seq(), Seq(), Seq(), Seq(Region(Seq(bb3, bb4))))

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
              Seq(
                Operation(
                  "test.op",
                  Seq(),
                  Seq(),
                  Seq(
                    Value(I32),
                    Value(I64),
                    Value(I32)
                  ),
                  Seq()
                )
              ),
              48
            ) =>
      }
    }
  }

  "TopLevel Tests2" should "Test full programs" in {
    withClue("Test 2: ") {
      parser.parseThis(
        text = "%0, %1, %2 = \"test.op\"() : () -> (i32, i64, i32)\n" +
          "\"test.op\"(%1, %0) : (i64, i32) -> ()",
        pattern = parser.TopLevel(_)
      ) should matchPattern {
        case Parsed.Success(
              Seq(
                Operation(
                  "test.op",
                  Seq(),
                  Seq(),
                  Seq(
                    r0,
                    r1,
                    r2
                  ),
                  Seq()
                ),
                Operation(
                  "test.op",
                  Seq(o0, o1),
                  Seq(),
                  Seq(),
                  Seq()
                )
              ),
              85
            ) if (o0 eq r1) && (o1 eq r0) =>
      }
    }
  }
}

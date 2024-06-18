package scair

import fastparse._, MultiLineWhitespace._
import scala.collection.mutable
import scala.util.{Try, Success, Failure}
import org.scalatest._
import Matchers._
import Parser._
import org.scalatest.prop.Tables.Table
import org.scalatest.prop.TableDrivenPropertyChecks.forAll

class ParserTest extends FlatSpec with BeforeAndAfter {
  var parser: Parser = new Parser

  before {
    parser = new Parser
  }

  "Digits - Unit Tests" should "parse correctly" in {
    withClue("Test 1: ") {
      parse("a", parser.Digit(_)) should matchPattern {
        case Parsed.Failure(_, _, _) =>
      }
    }
    withClue("Test 2: ") {
      parser.parseThis(
        " $ ! £ 4 1 ",
        parser.Digit(_)
      ) should matchPattern { case Parsed.Failure(_, _, _) => }
    }
    withClue("Test 3: ") {
      parser.parseThis("7", parser.Digit(_)) should matchPattern {
        case Parsed.Success("7", _) =>
      }
    }
  }

  val hexsucces = Table(
    ("input", "expected"),
    ("5", "5"),
    ("f", "f"),
    ("E", "E"),
    ("41", "4")
  )

  val hexfailures = Table(
    ("input", "expected"),
    ("G", ""),
    ("g", "")
  )

  forAll(hexsucces) { (input, expected) =>
    s"HexDigits - Unit Tests - ${input}" should "parse correctly" in {
      parser.parseThis(input, parser.HexDigit(_)) should matchPattern {
        case Parsed.Success(`expected`, _) =>
      }
    }
    forAll(hexfailures) { (input, expected) =>
      parser.parseThis(input, parser.HexDigit(_)) should matchPattern {
        case Parsed.Failure(`expected`, _, _) =>
      }
    }
  }

  "Letters - Unit Tests" should "parse correctly" in {
    withClue("Test 10: ") {
      parser.parseThis("a", parser.Letter(_)) should matchPattern {
        case Parsed.Success("a", _) =>
      }
    }
    withClue("Test 11: ") {
      parser.parseThis("G", parser.Letter(_)) should matchPattern {
        case Parsed.Success("G", _) =>
      }
    }
    withClue("Test 12: ") {
      parser.parseThis("4", parser.Letter(_)) should matchPattern {
        case Parsed.Failure(_, _, _) =>
      }
    }
  }

  "IdPuncts - Unit Tests" should "parse correctly" in {
    withClue("Test 13: ") {
      parser.parseThis("$", parser.IdPunct(_)) should matchPattern {
        case Parsed.Success("$", _) =>
      }
    }
    withClue("Test 14: ") {
      parser.parseThis(".", parser.IdPunct(_)) should matchPattern {
        case Parsed.Success(".", _) =>
      }
    }
    withClue("Test 15: ") {
      parser.parseThis("_", parser.IdPunct(_)) should matchPattern {
        case Parsed.Success("_", _) =>
      }
    }
    withClue("Test 16: ") {
      parser.parseThis("-", parser.IdPunct(_)) should matchPattern {
        case Parsed.Success("-", _) =>
      }
    }
    withClue("Test 17: ") {
      parser.parseThis("%", parser.IdPunct(_)) should matchPattern {
        case Parsed.Failure(_, _, _) =>
      }
    }
    withClue("Test 18: ") {
      parser.parseThis("£", parser.IdPunct(_)) should matchPattern {
        case Parsed.Failure(_, _, _) =>
      }
    }
    withClue("Test 19: ") {
      parser.parseThis("dfd", parser.IdPunct(_)) should matchPattern {
        case Parsed.Failure(_, _, _) =>
      }
    }
    withClue("Test 20: ") {
      parser.parseThis("0", parser.IdPunct(_)) should matchPattern {
        case Parsed.Failure(_, _, _) =>
      }
    }
  }

  "IntegerLiterals - Unit Tests" should "parse correctly" in {
    withClue("Test 21: ") {
      parser.parseThis(
        "123456789",
        parser.IntegerLiteral(_)
      ) should matchPattern { case Parsed.Success(123456789, _) => }
    }
    withClue("Test 22: ") {
      parser.parseThis(
        "1231f",
        parser.IntegerLiteral(_)
      ) should matchPattern { case Parsed.Success(1231, _) => }
    }
    withClue("Test 23: ") {
      parser.parseThis(
        "f1231",
        parser.IntegerLiteral(_)
      ) should matchPattern { case Parsed.Failure(_, _, _) => }
    }
    withClue("Test 24: ") {
      parser.parseThis(
        "0x0011ffff",
        parser.IntegerLiteral(_)
      ) should matchPattern { case Parsed.Success(0x0011ffff, _) => }
    }
    withClue("Test 25: ") {
      parser.parseThis(
        "0x0011gggg",
        parser.IntegerLiteral(_)
      ) should matchPattern { case Parsed.Success(0x0011, _) => }
    }
    withClue("Test 26: ") {
      parser.parseThis(
        "1xds%",
        parser.IntegerLiteral(_)
      ) should matchPattern { case Parsed.Success(1, _) => }
    }
    withClue("Test 27: ") {
      parser.parseThis(
        "0xgg",
        parser.IntegerLiteral(_)
      ) should matchPattern { case Parsed.Success(0, _) => }
    }
  }

  "DecimalLiterals - Unit Tests" should "parse correctly" in {
    withClue("Test 28: ") {
      parser.parseThis(
        "123456789",
        parser.DecimalLiteral(_)
      ) should matchPattern { case Parsed.Success(123456789, _) => }
    }
    withClue("Test 29: ") {
      parser.parseThis(
        "1231f",
        parser.DecimalLiteral(_)
      ) should matchPattern { case Parsed.Success(1231, _) => }
    }
    withClue("Test 30: ") {
      parser.parseThis(
        "f1231",
        parser.DecimalLiteral(_)
      ) should matchPattern { case Parsed.Failure(_, _, _) => }
    }
  }

  "HexadecimalLiterals - Unit Tests" should "parse correctly" in {
    withClue("Test 31: ") {
      parser.parseThis(
        "0x0011ffff",
        parser.HexadecimalLiteral(_)
      ) should matchPattern { case Parsed.Success(0x0011ffff, _) => }
    }
    withClue("Test 32: ") {
      parser.parseThis(
        "0x0011gggg",
        parser.HexadecimalLiteral(_)
      ) should matchPattern { case Parsed.Success(0x0011, _) => }
    }
    withClue("Test 33: ") {
      parser.parseThis(
        "1xds%",
        parser.HexadecimalLiteral(_)
      ) should matchPattern { case Parsed.Failure(_, _, _) => }
    }
    withClue("Test 34: ") {
      parser.parseThis(
        "0xgg",
        parser.HexadecimalLiteral(_)
      ) should matchPattern { case Parsed.Failure(_, _, _) => }
    }
  }

  "FloatLiterals - Unit Tests" should "parse correctly" in {
    withClue("Test 35: ") {
      parser.parseThis("1.0", parser.FloatLiteral(_)) should matchPattern {
        case Parsed.Success("1.0", _) =>
      }
    }
    withClue("Test 36: ") {
      parser.parseThis(
        "1.01242",
        parser.FloatLiteral(_)
      ) should matchPattern { case Parsed.Success("1.01242", _) => }
    }
    withClue("Test 37: ") {
      parser.parseThis(
        "993.013131",
        parser.FloatLiteral(_)
      ) should matchPattern { case Parsed.Success("993.013131", _) => }
    }
    withClue("Test 38: ") {
      parser.parseThis(
        "1.0e10",
        parser.FloatLiteral(_)
      ) should matchPattern { case Parsed.Success("1.0e10", _) => }
    }
    withClue("Test 39: ") {
      parser.parseThis(
        "1.0E10",
        parser.FloatLiteral(_)
      ) should matchPattern { case Parsed.Success("1.0E10", _) => }
    }
    withClue("Test 40: ") {
      parser.parseThis("1.", parser.FloatLiteral(_)) should matchPattern {
        case Parsed.Failure(_, _, _) =>
      }
    }
    withClue("Test 41: ") {
      parser.parseThis(
        "1.0E10",
        parser.FloatLiteral(_)
      ) should matchPattern { case Parsed.Success("1.0E10", _) => }
    }
    withClue("Test 42: ") {
      parser.parseThis(
        "1.0E10",
        parser.FloatLiteral(_)
      ) should matchPattern { case Parsed.Success("1.0E10", _) => }
    }
  }

  "StringLiterals - Unit Tests" should "parse correctly" in {
    withClue("Test 43: ") {
      parser.parseThis(
        "\"hello\"",
        parser.StringLiteral(_)
      ) should matchPattern { case Parsed.Success("hello", _) => }
    }
    withClue("Test 44: ") {
      parser.parseThis(
        "\"hello\"",
        parser.StringLiteral(_)
      ) should matchPattern { case Parsed.Success("hello", _) => }
    }
  }

  "ValueId - Unit Tests" should "parse correctly" in {
    withClue("Test 45: ") {
      parser.parseThis("%hello", parser.ValueId(_)) should matchPattern {
        case Parsed.Success("hello", _) =>
      }
    }
    withClue("Test 46: ") {
      parser.parseThis("%Ater", parser.ValueId(_)) should matchPattern {
        case Parsed.Success("Ater", _) =>
      }
    }
    withClue("Test 47: ") {
      parser.parseThis("%312321", parser.ValueId(_)) should matchPattern {
        case Parsed.Success("312321", _) =>
      }
    }
    withClue("Test 48: ") {
      parser.parseThis("%Ater", parser.ValueId(_)) should matchPattern {
        case Parsed.Success("Ater", _) =>
      }
    }
    withClue("Test 49: ") {
      parser.parseThis("%$$$$$", parser.ValueId(_)) should matchPattern {
        case Parsed.Success("$$$$$", _) =>
      }
    }
    withClue("Test 50: ") {
      parser.parseThis("%_-_-_", parser.ValueId(_)) should matchPattern {
        case Parsed.Success("_-_-_", _) =>
      }
    }
    withClue("Test 51: ") {
      parser.parseThis("%3asada", parser.ValueId(_)) should matchPattern {
        case Parsed.Success("3", _) =>
      }
    }
    withClue("Test 52: ") {
      parser.parseThis("% hello", parser.ValueId(_)) should matchPattern {
        case Parsed.Failure(_, _, _) =>
      }
    }
  }

  "OpResultList - Unit Tests" should "parse correctly" in {
    withClue("Test 53: ") {
      parser.parseThis(
        "%0, %1, %2 =",
        parser.OpResultList(_)
      ) should matchPattern { case Parsed.Success(List("0", "1", "2"), _) => }
    }
    withClue("Test 54: ") {
      parser.parseThis(
        "%0   ,    %1   ,   %2  =   ",
        parser.OpResultList(_)
      ) should matchPattern { case Parsed.Success(List("0", "1", "2"), _) => }
    }
    withClue("Test 55: ") {
      parser.parseThis(
        "%0,%1,%2=",
        parser.OpResultList(_)
      ) should matchPattern { case Parsed.Success(List("0", "1", "2"), _) => }
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
                      Value(Attribute("i32")),
                      Value(Attribute("i64")),
                      Value(Attribute("i32"))
                    ),
                    Seq()
                  ),
                  Operation(
                    "test.op",
                    Seq(Value(Attribute("i64")), Value(Attribute("i32"))),
                    Seq(),
                    Seq(),
                    Seq()
                  )
                ),
                Seq(Value(Attribute("i32")))
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
                          Value(Attribute("i32")),
                          Value(Attribute("i64")),
                          Value(Attribute("i32"))
                        ),
                        Seq()
                      ),
                      Operation(
                        "test.op",
                        Seq(Value(Attribute("i64")), Value(Attribute("i32"))),
                        Seq(),
                        Seq(),
                        Seq()
                      )
                    ),
                    Seq(Value(Attribute("i32")))
                  ),
                  Block(
                    Seq(
                      Operation(
                        "test.op",
                        Seq(),
                        Seq(),
                        Seq(
                          Value(Attribute("i32")),
                          Value(Attribute("i64")),
                          Value(Attribute("i32"))
                        ),
                        Seq()
                      ),
                      Operation(
                        "test.op",
                        Seq(
                          Value(Attribute("i64")),
                          Value(Attribute("i32"))
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
                    Value(Attribute("i32")),
                    Value(Attribute("i64")),
                    Value(Attribute("i32"))
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

  // [ ] - test successors
  // [ ] - test re-format tests
  // [ ] - test re-format tests
}

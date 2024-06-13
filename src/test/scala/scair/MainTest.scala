package scair

import fastparse._, MultiLineWhitespace._
import scala.collection.mutable
import scala.util.{Try, Success, Failure}
import org.scalatest._
import Matchers._
import Parser._

class MainTest extends FlatSpec {

  val parser = new Parser

  "Digits - Unit Tests" should "parse correctly" in {
    withClue("Test 1: ") {
      parser.testParse("a", parser.Digit(_)) should matchPattern {
        case Parsed.Failure(_, _, _) =>
      }
    }
    withClue("Test 2: ") {
      parser.testParse(
        " $ ! £ 4 1 ",
        parser.Digit(_)
      ) should matchPattern { case Parsed.Failure(_, _, _) => }
    }
    withClue("Test 3: ") {
      parser.testParse("7", parser.Digit(_)) should matchPattern {
        case Parsed.Success("7", _) =>
      }
    }
  }

  "HexDigits - Unit Tests" should "parse correctly" in {
    withClue("Test 4: ") {
      parser.testParse("5", parser.HexDigit(_)) should matchPattern {
        case Parsed.Success("5", _) =>
      }
    }
    withClue("Test 5: ") {
      parser.testParse("f", parser.HexDigit(_)) should matchPattern {
        case Parsed.Success("f", _) =>
      }
    }
    withClue("Test 6: ") {
      parser.testParse("E", parser.HexDigit(_)) should matchPattern {
        case Parsed.Success("E", _) =>
      }
    }
    withClue("Test 7: ") {
      parser.testParse("G", parser.HexDigit(_)) should matchPattern {
        case Parsed.Failure(_, _, _) =>
      }
    }
    withClue("Test 8: ") {
      parser.testParse("g", parser.HexDigit(_)) should matchPattern {
        case Parsed.Failure(_, _, _) =>
      }
    }
    withClue("Test 9: ") {
      parser.testParse("41", parser.HexDigit(_)) should matchPattern {
        case Parsed.Success("4", _) =>
      }
    }
  }

  "Letters - Unit Tests" should "parse correctly" in {
    withClue("Test 10: ") {
      parser.testParse("a", parser.Letter(_)) should matchPattern {
        case Parsed.Success("a", _) =>
      }
    }
    withClue("Test 11: ") {
      parser.testParse("G", parser.Letter(_)) should matchPattern {
        case Parsed.Success("G", _) =>
      }
    }
    withClue("Test 12: ") {
      parser.testParse("4", parser.Letter(_)) should matchPattern {
        case Parsed.Failure(_, _, _) =>
      }
    }
  }

  "IdPuncts - Unit Tests" should "parse correctly" in {
    withClue("Test 13: ") {
      parser.testParse("$", parser.IdPunct(_)) should matchPattern {
        case Parsed.Success("$", _) =>
      }
    }
    withClue("Test 14: ") {
      parser.testParse(".", parser.IdPunct(_)) should matchPattern {
        case Parsed.Success(".", _) =>
      }
    }
    withClue("Test 15: ") {
      parser.testParse("_", parser.IdPunct(_)) should matchPattern {
        case Parsed.Success("_", _) =>
      }
    }
    withClue("Test 16: ") {
      parser.testParse("-", parser.IdPunct(_)) should matchPattern {
        case Parsed.Success("-", _) =>
      }
    }
    withClue("Test 17: ") {
      parser.testParse("%", parser.IdPunct(_)) should matchPattern {
        case Parsed.Failure(_, _, _) =>
      }
    }
    withClue("Test 18: ") {
      parser.testParse("£", parser.IdPunct(_)) should matchPattern {
        case Parsed.Failure(_, _, _) =>
      }
    }
    withClue("Test 19: ") {
      parser.testParse("dfd", parser.IdPunct(_)) should matchPattern {
        case Parsed.Failure(_, _, _) =>
      }
    }
    withClue("Test 20: ") {
      parser.testParse("0", parser.IdPunct(_)) should matchPattern {
        case Parsed.Failure(_, _, _) =>
      }
    }
  }

  "IntegerLiterals - Unit Tests" should "parse correctly" in {
    withClue("Test 21: ") {
      parser.testParse(
        "123456789",
        parser.IntegerLiteral(_)
      ) should matchPattern { case Parsed.Success(123456789, _) => }
    }
    withClue("Test 22: ") {
      parser.testParse(
        "1231f",
        parser.IntegerLiteral(_)
      ) should matchPattern { case Parsed.Success(1231, _) => }
    }
    withClue("Test 23: ") {
      parser.testParse(
        "f1231",
        parser.IntegerLiteral(_)
      ) should matchPattern { case Parsed.Failure(_, _, _) => }
    }
    withClue("Test 24: ") {
      parser.testParse(
        "0x0011ffff",
        parser.IntegerLiteral(_)
      ) should matchPattern { case Parsed.Success(0x0011ffff, _) => }
    }
    withClue("Test 25: ") {
      parser.testParse(
        "0x0011gggg",
        parser.IntegerLiteral(_)
      ) should matchPattern { case Parsed.Success(0x0011, _) => }
    }
    withClue("Test 26: ") {
      parser.testParse(
        "1xds%",
        parser.IntegerLiteral(_)
      ) should matchPattern { case Parsed.Success(1, _) => }
    }
    withClue("Test 27: ") {
      parser.testParse(
        "0xgg",
        parser.IntegerLiteral(_)
      ) should matchPattern { case Parsed.Success(0, _) => }
    }
  }

  "DecimalLiterals - Unit Tests" should "parse correctly" in {
    withClue("Test 28: ") {
      parser.testParse(
        "123456789",
        parser.DecimalLiteral(_)
      ) should matchPattern { case Parsed.Success(123456789, _) => }
    }
    withClue("Test 29: ") {
      parser.testParse(
        "1231f",
        parser.DecimalLiteral(_)
      ) should matchPattern { case Parsed.Success(1231, _) => }
    }
    withClue("Test 30: ") {
      parser.testParse(
        "f1231",
        parser.DecimalLiteral(_)
      ) should matchPattern { case Parsed.Failure(_, _, _) => }
    }
  }

  "HexadecimalLiterals - Unit Tests" should "parse correctly" in {
    withClue("Test 31: ") {
      parser.testParse(
        "0x0011ffff",
        parser.HexadecimalLiteral(_)
      ) should matchPattern { case Parsed.Success(0x0011ffff, _) => }
    }
    withClue("Test 32: ") {
      parser.testParse(
        "0x0011gggg",
        parser.HexadecimalLiteral(_)
      ) should matchPattern { case Parsed.Success(0x0011, _) => }
    }
    withClue("Test 33: ") {
      parser.testParse(
        "1xds%",
        parser.HexadecimalLiteral(_)
      ) should matchPattern { case Parsed.Failure(_, _, _) => }
    }
    withClue("Test 34: ") {
      parser.testParse(
        "0xgg",
        parser.HexadecimalLiteral(_)
      ) should matchPattern { case Parsed.Failure(_, _, _) => }
    }
  }

  "FloatLiterals - Unit Tests" should "parse correctly" in {
    withClue("Test 35: ") {
      parser.testParse("1.0", parser.FloatLiteral(_)) should matchPattern {
        case Parsed.Success("1.0", _) =>
      }
    }
    withClue("Test 36: ") {
      parser.testParse(
        "1.01242",
        parser.FloatLiteral(_)
      ) should matchPattern { case Parsed.Success("1.01242", _) => }
    }
    withClue("Test 37: ") {
      parser.testParse(
        "993.013131",
        parser.FloatLiteral(_)
      ) should matchPattern { case Parsed.Success("993.013131", _) => }
    }
    withClue("Test 38: ") {
      parser.testParse(
        "1.0e10",
        parser.FloatLiteral(_)
      ) should matchPattern { case Parsed.Success("1.0e10", _) => }
    }
    withClue("Test 39: ") {
      parser.testParse(
        "1.0E10",
        parser.FloatLiteral(_)
      ) should matchPattern { case Parsed.Success("1.0E10", _) => }
    }
    withClue("Test 40: ") {
      parser.testParse("1.", parser.FloatLiteral(_)) should matchPattern {
        case Parsed.Failure(_, _, _) =>
      }
    }
    withClue("Test 41: ") {
      parser.testParse(
        "1.0E10",
        parser.FloatLiteral(_)
      ) should matchPattern { case Parsed.Success("1.0E10", _) => }
    }
    withClue("Test 42: ") {
      parser.testParse(
        "1.0E10",
        parser.FloatLiteral(_)
      ) should matchPattern { case Parsed.Success("1.0E10", _) => }
    }
  }

  "StringLiterals - Unit Tests" should "parse correctly" in {
    withClue("Test 43: ") {
      parser.testParse(
        "\"hello\"",
        parser.StringLiteral(_)
      ) should matchPattern { case Parsed.Success("hello", _) => }
    }
    withClue("Test 44: ") {
      parser.testParse(
        "\"hello\"",
        parser.StringLiteral(_)
      ) should matchPattern { case Parsed.Success("hello", _) => }
    }
  }

  "ValueId - Unit Tests" should "parse correctly" in {
    withClue("Test 45: ") {
      parser.testParse("%hello", parser.ValueId(_)) should matchPattern {
        case Parsed.Success("hello", _) =>
      }
    }
    withClue("Test 46: ") {
      parser.testParse("%Ater", parser.ValueId(_)) should matchPattern {
        case Parsed.Success("Ater", _) =>
      }
    }
    withClue("Test 47: ") {
      parser.testParse("%312321", parser.ValueId(_)) should matchPattern {
        case Parsed.Success("312321", _) =>
      }
    }
    withClue("Test 48: ") {
      parser.testParse("%Ater", parser.ValueId(_)) should matchPattern {
        case Parsed.Success("Ater", _) =>
      }
    }
    withClue("Test 49: ") {
      parser.testParse("%$$$$$", parser.ValueId(_)) should matchPattern {
        case Parsed.Success("$$$$$", _) =>
      }
    }
    withClue("Test 50: ") {
      parser.testParse("%_-_-_", parser.ValueId(_)) should matchPattern {
        case Parsed.Success("_-_-_", _) =>
      }
    }
    withClue("Test 51: ") {
      parser.testParse("%3asada", parser.ValueId(_)) should matchPattern {
        case Parsed.Success("3", _) =>
      }
    }
    withClue("Test 52: ") {
      parser.testParse("% hello", parser.ValueId(_)) should matchPattern {
        case Parsed.Failure(_, _, _) =>
      }
    }
  }

  "OpResultList - Unit Tests" should "parse correctly" in {
    withClue("Test 53: ") {
      parser.testParse(
        "%0, %1, %2 =",
        parser.OpResultList(_)
      ) should matchPattern { case Parsed.Success(List("0", "1", "2"), _) => }
    }
    withClue("Test 54: ") {
      parser.testParse(
        "%0   ,    %1   ,   %2  =   ",
        parser.OpResultList(_)
      ) should matchPattern { case Parsed.Success(List("0", "1", "2"), _) => }
    }
    withClue("Test 55: ") {
      parser.testParse(
        "%0,%1,%2=",
        parser.OpResultList(_)
      ) should matchPattern { case Parsed.Success(List("0", "1", "2"), _) => }
    }
  }

  "Block - Unit Tests" should "parse correctly" in {
    withClue("Test 1: ") {
      parser.testParse(
        text =
          "^bb0(%5: i32):\n" + "%0, %1, %2 = \"test.op\"() : () -> (i32, i64, i32)\n" +
            "\"test.op\"(%1, %0) : (i64, i32) -> ()",
        parser = parser.Block(_)
      ) should matchPattern {
        case Parsed.Success(
              Block(
                Seq(
                  Operation(
                    "test.op",
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
      parser.testParse(
        text =
          "{^bb0(%5: i32):\n" + "%0, %1, %2 = \"test.op\"() : () -> (i32, i64, i32)\n" +
            "\"test.op\"(%1, %0) : (i64, i32) -> ()" + "^bb1(%4: i32):\n" + "%7, %8, %9 = \"test.op\"() : () -> (i32, i64, i32)\n" +
            "\"test.op\"(%8, %7) : (i64, i32) -> ()" + "}",
        parser = parser.Region(_)
      ) should matchPattern {
        case Parsed.Success(
              Region(
                Seq(
                  Block(
                    Seq(
                      Operation(
                        "test.op",
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

    withClue("Test 2: ") {
      val exception = intercept[Exception](
        parser.testParse(
          text =
            "{^bb0(%5: i32):\n" + "%0, %1, %2 = \"test.op\"() : () -> (i32, i64, i32)\n" +
              "\"test.op\"(%1, %0) : (i64, i32) -> ()" + "^bb0(%4: i32):\n" + "%7, %8, %9 = \"test.op\"() : () -> (i32, i64, i32)\n" +
              "\"test.op\"(%8, %7) : (i64, i32) -> ()" + "}",
          parser = parser.Region(_)
        )
      ).getMessage shouldBe "Block cannot be defined twice within the same scope - ^bb0"
    }
  }

  "TopLevel Tests" should "Test full programs" in {
    withClue("Test 1: ") {
      parser.testParse(
        text = "%0, %1, %2 = \"test.op\"() : () -> (i32, i64, i32)",
        parser = parser.TopLevel(_)
      ) should matchPattern {
        case Parsed.Success(
              Seq(
                Operation(
                  "test.op",
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
    withClue("Test 2: ") {
      parser.testParse(
        text = "%0, %1, %2 = \"test.op\"() : () -> (i32, i64, i32)\n" +
          "\"test.op\"(%1, %0) : (i64, i32) -> ()",
        parser = parser.TopLevel(_)
      ) should matchPattern {
        case Parsed.Success(
              Seq(
                Operation(
                  "test.op",
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
                  Seq()
                )
              ),
              85
            ) if (o0 eq r1) && (o1 eq r0) =>
      }
    }
  }
}

/*

"Unit Tests" should "test for obvious" in {
    true should be (!false)
    parser.testParse("a", parser.Digit(_)) should matchPattern {case Parsed.Failure(_, _, _) => }

  }

 */

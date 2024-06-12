package scair

import fastparse._, MultiLineWhitespace._
import scala.collection.mutable
import scala.util.{Try, Success, Failure}
import org.scalatest._
import Matchers._
import Parser._

class MainTest extends FlatSpec {

  val testContext: Parser.Context =
    new Parser.Context(localValueMap = mutable.Map.empty[String, Value])

  "Digits - Unit Tests" should "parse correctly" in {
    withClue("Test 1: ") {
      testContext.testParse("a", Parser.Digit(_)) should matchPattern {
        case Parsed.Failure(_, _, _) =>
      }
    }
    withClue("Test 2: ") {
      testContext.testParse(
        " $ ! £ 4 1 ",
        Parser.Digit(_)
      ) should matchPattern { case Parsed.Failure(_, _, _) => }
    }
    withClue("Test 3: ") {
      testContext.testParse("7", Parser.Digit(_)) should matchPattern {
        case Parsed.Success("7", _) =>
      }
    }
  }

  "HexDigits - Unit Tests" should "parse correctly" in {
    withClue("Test 4: ") {
      testContext.testParse("5", Parser.HexDigit(_)) should matchPattern {
        case Parsed.Success("5", _) =>
      }
    }
    withClue("Test 5: ") {
      testContext.testParse("f", Parser.HexDigit(_)) should matchPattern {
        case Parsed.Success("f", _) =>
      }
    }
    withClue("Test 6: ") {
      testContext.testParse("E", Parser.HexDigit(_)) should matchPattern {
        case Parsed.Success("E", _) =>
      }
    }
    withClue("Test 7: ") {
      testContext.testParse("G", Parser.HexDigit(_)) should matchPattern {
        case Parsed.Failure(_, _, _) =>
      }
    }
    withClue("Test 8: ") {
      testContext.testParse("g", Parser.HexDigit(_)) should matchPattern {
        case Parsed.Failure(_, _, _) =>
      }
    }
    withClue("Test 9: ") {
      testContext.testParse("41", Parser.HexDigit(_)) should matchPattern {
        case Parsed.Success("4", _) =>
      }
    }
  }

  "Letters - Unit Tests" should "parse correctly" in {
    withClue("Test 10: ") {
      testContext.testParse("a", Parser.Letter(_)) should matchPattern {
        case Parsed.Success("a", _) =>
      }
    }
    withClue("Test 11: ") {
      testContext.testParse("G", Parser.Letter(_)) should matchPattern {
        case Parsed.Success("G", _) =>
      }
    }
    withClue("Test 12: ") {
      testContext.testParse("4", Parser.Letter(_)) should matchPattern {
        case Parsed.Failure(_, _, _) =>
      }
    }
  }

  "IdPuncts - Unit Tests" should "parse correctly" in {
    withClue("Test 13: ") {
      testContext.testParse("$", Parser.IdPunct(_)) should matchPattern {
        case Parsed.Success("$", _) =>
      }
    }
    withClue("Test 14: ") {
      testContext.testParse(".", Parser.IdPunct(_)) should matchPattern {
        case Parsed.Success(".", _) =>
      }
    }
    withClue("Test 15: ") {
      testContext.testParse("_", Parser.IdPunct(_)) should matchPattern {
        case Parsed.Success("_", _) =>
      }
    }
    withClue("Test 16: ") {
      testContext.testParse("-", Parser.IdPunct(_)) should matchPattern {
        case Parsed.Success("-", _) =>
      }
    }
    withClue("Test 17: ") {
      testContext.testParse("%", Parser.IdPunct(_)) should matchPattern {
        case Parsed.Failure(_, _, _) =>
      }
    }
    withClue("Test 18: ") {
      testContext.testParse("£", Parser.IdPunct(_)) should matchPattern {
        case Parsed.Failure(_, _, _) =>
      }
    }
    withClue("Test 19: ") {
      testContext.testParse("dfd", Parser.IdPunct(_)) should matchPattern {
        case Parsed.Failure(_, _, _) =>
      }
    }
    withClue("Test 20: ") {
      testContext.testParse("0", Parser.IdPunct(_)) should matchPattern {
        case Parsed.Failure(_, _, _) =>
      }
    }
  }

  "IntegerLiterals - Unit Tests" should "parse correctly" in {
    withClue("Test 21: ") {
      testContext.testParse(
        "123456789",
        Parser.IntegerLiteral(_)
      ) should matchPattern { case Parsed.Success(123456789, _) => }
    }
    withClue("Test 22: ") {
      testContext.testParse(
        "1231f",
        Parser.IntegerLiteral(_)
      ) should matchPattern { case Parsed.Success(1231, _) => }
    }
    withClue("Test 23: ") {
      testContext.testParse(
        "f1231",
        Parser.IntegerLiteral(_)
      ) should matchPattern { case Parsed.Failure(_, _, _) => }
    }
    withClue("Test 24: ") {
      testContext.testParse(
        "0x0011ffff",
        Parser.IntegerLiteral(_)
      ) should matchPattern { case Parsed.Success(0x0011ffff, _) => }
    }
    withClue("Test 25: ") {
      testContext.testParse(
        "0x0011gggg",
        Parser.IntegerLiteral(_)
      ) should matchPattern { case Parsed.Success(0x0011, _) => }
    }
    withClue("Test 26: ") {
      testContext.testParse(
        "1xds%",
        Parser.IntegerLiteral(_)
      ) should matchPattern { case Parsed.Success(1, _) => }
    }
    withClue("Test 27: ") {
      testContext.testParse(
        "0xgg",
        Parser.IntegerLiteral(_)
      ) should matchPattern { case Parsed.Success(0, _) => }
    }
  }

  "DecimalLiterals - Unit Tests" should "parse correctly" in {
    withClue("Test 28: ") {
      testContext.testParse(
        "123456789",
        Parser.DecimalLiteral(_)
      ) should matchPattern { case Parsed.Success(123456789, _) => }
    }
    withClue("Test 29: ") {
      testContext.testParse(
        "1231f",
        Parser.DecimalLiteral(_)
      ) should matchPattern { case Parsed.Success(1231, _) => }
    }
    withClue("Test 30: ") {
      testContext.testParse(
        "f1231",
        Parser.DecimalLiteral(_)
      ) should matchPattern { case Parsed.Failure(_, _, _) => }
    }
  }

  "HexadecimalLiterals - Unit Tests" should "parse correctly" in {
    withClue("Test 31: ") {
      testContext.testParse(
        "0x0011ffff",
        Parser.HexadecimalLiteral(_)
      ) should matchPattern { case Parsed.Success(0x0011ffff, _) => }
    }
    withClue("Test 32: ") {
      testContext.testParse(
        "0x0011gggg",
        Parser.HexadecimalLiteral(_)
      ) should matchPattern { case Parsed.Success(0x0011, _) => }
    }
    withClue("Test 33: ") {
      testContext.testParse(
        "1xds%",
        Parser.HexadecimalLiteral(_)
      ) should matchPattern { case Parsed.Failure(_, _, _) => }
    }
    withClue("Test 34: ") {
      testContext.testParse(
        "0xgg",
        Parser.HexadecimalLiteral(_)
      ) should matchPattern { case Parsed.Failure(_, _, _) => }
    }
  }

  "FloatLiterals - Unit Tests" should "parse correctly" in {
    withClue("Test 35: ") {
      testContext.testParse("1.0", Parser.FloatLiteral(_)) should matchPattern {
        case Parsed.Success("1.0", _) =>
      }
    }
    withClue("Test 36: ") {
      testContext.testParse(
        "1.01242",
        Parser.FloatLiteral(_)
      ) should matchPattern { case Parsed.Success("1.01242", _) => }
    }
    withClue("Test 37: ") {
      testContext.testParse(
        "993.013131",
        Parser.FloatLiteral(_)
      ) should matchPattern { case Parsed.Success("993.013131", _) => }
    }
    withClue("Test 38: ") {
      testContext.testParse(
        "1.0e10",
        Parser.FloatLiteral(_)
      ) should matchPattern { case Parsed.Success("1.0e10", _) => }
    }
    withClue("Test 39: ") {
      testContext.testParse(
        "1.0E10",
        Parser.FloatLiteral(_)
      ) should matchPattern { case Parsed.Success("1.0E10", _) => }
    }
    withClue("Test 40: ") {
      testContext.testParse("1.", Parser.FloatLiteral(_)) should matchPattern {
        case Parsed.Failure(_, _, _) =>
      }
    }
    withClue("Test 41: ") {
      testContext.testParse(
        "1.0E10",
        Parser.FloatLiteral(_)
      ) should matchPattern { case Parsed.Success("1.0E10", _) => }
    }
    withClue("Test 42: ") {
      testContext.testParse(
        "1.0E10",
        Parser.FloatLiteral(_)
      ) should matchPattern { case Parsed.Success("1.0E10", _) => }
    }
  }

  "StringLiterals - Unit Tests" should "parse correctly" in {
    withClue("Test 43: ") {
      testContext.testParse(
        "\"hello\"",
        Parser.StringLiteral(_)
      ) should matchPattern { case Parsed.Success("hello", _) => }
    }
    withClue("Test 44: ") {
      testContext.testParse(
        "\"hello\"",
        Parser.StringLiteral(_)
      ) should matchPattern { case Parsed.Success("hello", _) => }
    }
  }

  "ValueId - Unit Tests" should "parse correctly" in {
    withClue("Test 45: ") {
      testContext.testParse("%hello", Parser.ValueId(_)) should matchPattern {
        case Parsed.Success("hello", _) =>
      }
    }
    withClue("Test 46: ") {
      testContext.testParse("%Ater", Parser.ValueId(_)) should matchPattern {
        case Parsed.Success("Ater", _) =>
      }
    }
    withClue("Test 47: ") {
      testContext.testParse("%312321", Parser.ValueId(_)) should matchPattern {
        case Parsed.Success("312321", _) =>
      }
    }
    withClue("Test 48: ") {
      testContext.testParse("%Ater", Parser.ValueId(_)) should matchPattern {
        case Parsed.Success("Ater", _) =>
      }
    }
    withClue("Test 49: ") {
      testContext.testParse("%$$$$$", Parser.ValueId(_)) should matchPattern {
        case Parsed.Success("$$$$$", _) =>
      }
    }
    withClue("Test 50: ") {
      testContext.testParse("%_-_-_", Parser.ValueId(_)) should matchPattern {
        case Parsed.Success("_-_-_", _) =>
      }
    }
    withClue("Test 51: ") {
      testContext.testParse("%3asada", Parser.ValueId(_)) should matchPattern {
        case Parsed.Success("3", _) =>
      }
    }
    withClue("Test 52: ") {
      testContext.testParse("% hello", Parser.ValueId(_)) should matchPattern {
        case Parsed.Failure(_, _, _) =>
      }
    }
  }

  "OpResultList - Unit Tests" should "parse correctly" in {
    withClue("Test 53: ") {
      testContext.testParse(
        "%0, %1, %2 =",
        Parser.OpResultList(_)
      ) should matchPattern { case Parsed.Success(List("0", "1", "2"), _) => }
    }
    withClue("Test 54: ") {
      testContext.testParse(
        "%0   ,    %1   ,   %2  =   ",
        Parser.OpResultList(_)
      ) should matchPattern { case Parsed.Success(List("0", "1", "2"), _) => }
    }
    withClue("Test 55: ") {
      testContext.testParse(
        "%0,%1,%2=",
        Parser.OpResultList(_)
      ) should matchPattern { case Parsed.Success(List("0", "1", "2"), _) => }
    }
  }

  "TopLevel Tests" should "Test full programs" in {
    withClue("Test 1: ") {
      testContext.testParse(
        text = "%0, %1, %2 = \"test.op\"() : () -> (i32, i64, i32)",
        parser = TopLevel(_)
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
                  )
                )
              ),
              48
            ) =>
      }
    }
    withClue("Test 2: ") {
      testContext.testParse(
        text = "%0, %1, %2 = \"test.op\"() : () -> (i32, i64, i32)\n" +
          "\"test.op\"(%1, %0) : (i64, i32) -> ()",
        parser = TopLevel(_)
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
                  )
                ),
                Operation(
                  "test.op",
                  Seq(Value(Attribute("i64")), Value(Attribute("i32"))),
                  Seq()
                )
              ),
              85
            ) =>
      }
    }
  }
}

/*

"Unit Tests" should "test for obvious" in {
    true should be (!false)
    testContext.testParse("a", Parser.Digit(_)) should matchPattern {case Parsed.Failure(_, _, _) => }

  }

 */

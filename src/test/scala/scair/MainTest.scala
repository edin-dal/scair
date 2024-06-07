package scair

import fastparse._, MultiLineWhitespace._
import org.scalatest._
import Matchers._
import Parser._

class MainTest extends FlatSpec {

  "Main" should "test for obvious" in {
    true should be (!false)
  }
    val indent = "  "    

    var parser = Parser
    val Parsed.Failure(_, _, _) = parser.parse("a", Digit(_))
    // val Parsed.Failure(_, _, _) = parse(" $ ! £ 4 1 ", Digit(_))
    // val Parsed.Success("7", _) = parse("7", Digit(_))
    // println(indent + "Digits - OK")
    // // HexDigits
    // val Parsed.Success("5", _) = parse("5", HexDigit(_))
    // val Parsed.Success("f", _) = parse("f", HexDigit(_))
    // val Parsed.Success("E", _) = parse("E", HexDigit(_))
    // val Parsed.Failure(_, _, _) = parse("G", HexDigit(_))
    // val Parsed.Failure(_, _, _) = parse("g", HexDigit(_))
    // val Parsed.Success("4", _) = parse("41", HexDigit(_))
    // println(indent + "HexDigits - OK")
    // // Letters
    // val Parsed.Success("a", _) = parse("a", Letter(_))
    // val Parsed.Success("G", _) = parse("G", Letter(_))
    // val Parsed.Failure(_, _, _) = parse("4", Letter(_))
    // println(indent + "Letters - OK")
    // // IdPuncts
    // val Parsed.Success("$", _) = parse("$", IdPunct(_))
    // val Parsed.Success(".", _) = parse(".", IdPunct(_))
    // val Parsed.Success("_", _) = parse("_", IdPunct(_))
    // val Parsed.Success("-", _) = parse("-", IdPunct(_))
    // val Parsed.Failure(_, _, _) = parse("%", IdPunct(_))
    // val Parsed.Failure(_, _, _) = parse("£", IdPunct(_))
    // val Parsed.Failure(_, _, _) = parse("dfd", IdPunct(_))
    // val Parsed.Failure(_, _, _) = parse("0", IdPunct(_))
    // println(indent + "IdPunct - OK")
    // // IntegerLiterals
    // val Parsed.Success(123456789, _) = parse("123456789", IntegerLiteral(_))
    // val Parsed.Success(1231, _) = parse("1231f", IntegerLiteral(_))
    // val Parsed.Failure(_, _, _) = parse("f1231", IntegerLiteral(_))
    // val Parsed.Success(0x0011ffff, _) = parse("0x0011ffff", IntegerLiteral(_))
    // val Parsed.Success(0x0011, _) = parse("0x0011gggg", IntegerLiteral(_))
    // val Parsed.Success(1, _) = parse("1xds%", IntegerLiteral(_))
    // val Parsed.Success(0, _) = parse("0xgg", IntegerLiteral(_))
    // println(indent + "IntegerLiteral - OK")
    // // DecimalLiteral
    // val Parsed.Success(123456789, _) = parse("123456789", DecimalLiteral(_))
    // val Parsed.Success(1231, _) = parse("1231f", DecimalLiteral(_))
    // val Parsed.Failure(_, _, _) = parse("f1231", DecimalLiteral(_))
    // println(indent + "DecimalLiteral - OK")
    // // HexadecimalLiteral
    // val Parsed.Success(0x0011ffff, _) = parse("0x0011ffff", HexadecimalLiteral(_))
    // val Parsed.Success(0x0011, _) = parse("0x0011gggg", HexadecimalLiteral(_))
    // val Parsed.Failure(_, _, _) = parse("1xds%", HexadecimalLiteral(_))
    // val Parsed.Failure(_, _, _) = parse("0xgg", HexadecimalLiteral(_))
    // println(indent + "HexadecimalLiteral - OK")
    // // FloatLiteral
    // val Parsed.Success("1.0", _) = parse("1.0", FloatLiteral(_))
    // val Parsed.Success("1.01242", _) = parse("1.01242", FloatLiteral(_))
    // val Parsed.Success("993.013131", _) = parse("993.013131", FloatLiteral(_))
    // val Parsed.Success("1.0e10", _) = parse("1.0e10", FloatLiteral(_))
    // val Parsed.Success("1.0E10", _) = parse("1.0E10", FloatLiteral(_))
    // val Parsed.Failure(_, _, _) = parse("1.", FloatLiteral(_))
    // val Parsed.Success("1.0E10", _) = parse("1.0E10", FloatLiteral(_))
    // val Parsed.Success("1.0E10", _) = parse("1.0E10", FloatLiteral(_))
    // println(indent + "FloatLiteral - OK")
    // // StringLiteral
    // val Parsed.Success("hello", _) = parse("\"hello\"", StringLiteral(_))
    // val Parsed.Success("hello", _) = parse("\"hello\"", StringLiteral(_))
    // println(indent + "StringLiteral - OK")


    // println("IDENTIFIERS")
    // // ValueId
    // val Parsed.Success("hello", _) = parse("%hello", ValueId(_))
    // val Parsed.Success("Ater", _) = parse("%Ater", ValueId(_))
    // val Parsed.Success("312321", _) = parse("%312321", ValueId(_))
    // val Parsed.Success("Ater", _) = parse("%Ater", ValueId(_))
    // val Parsed.Success("$$$$$", _) = parse("%$$$$$", ValueId(_))
    // val Parsed.Success("_-_-_", _) = parse("%_-_-_", ValueId(_))
    // val Parsed.Success("3", _) = parse("%3asada", ValueId(_))
    // val Parsed.Failure(_, _, _) = parse("% hello", ValueId(_))
    // println(indent + "ValueId - OK")
    // //


    // println("OPERATIONS")
    // // OpResultList
    // println(parse("%0, %1, %2 =", OpResultList(_)))
    // // val Parsed.Success(Seq("%0", "%1", "%2"), _) = parse("%0, %1, %2 =", OpResultList(_))
    // // val Parsed.Success(Seq("%0", "%1", "%2"), _) = parse("%0   ,    %1   ,   %2  =   ", OpResultList(_))
    // // val Parsed.Success(Seq("%0", "%1", "%2"), _) = parse("%0,%1,%2=", OpResultList(_))
    // println(indent + "OpResultList - OK")

    // println("TOP LEVEL PRODUCTION")

    // println("----In-Progress----")
    // println(parse("%0, %1, %2 = \"test.op\"() : () -> (i32, i64, i32)", TopLevel(_)))
    // println(parse("%0, %1, %2 = \"test.op\"() : () -> (i32, i64, i32)\n" +
    //               "\"test.op\"(%1, %0) : (i32, i32) -> ()", TopLevel(_))) 
    // println(parse("\"test.op\"(%1, %0) : (i64, i32) -> ()", OperationPat(_))) 
}
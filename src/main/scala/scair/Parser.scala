package scair

import fastparse._, NoWhitespace._

object Parser {

    ///////////////////
    // COMMON SYNTAX // 
    ///////////////////

    // digit     ::= [0-9]
    // hex_digit ::= [0-9a-fA-F]
    // letter    ::= [a-zA-Z]
    // id-punct  ::= [$._-]

    // integer-literal ::= decimal-literal | hexadecimal-literal
    // decimal-literal ::= digit+
    // hexadecimal-literal ::= `0x` hex_digit+
    // float-literal ::= [-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?
    // string-literal  ::= `"` [^"\n\f\v\r]* `"`

    def Digit[$: P] = P( CharIn("0-9").! )
    def HexDigit[$: P] = P( CharIn("0-9a-fA-F").! )
    def Letter[$: P] = P( CharIn("a-zA-Z").! )
    def IdPunct[$: P] = P( CharIn("$._\\-").! )

    def IntegerLiteral[$: P] = P( HexadecimalLiteral | DecimalLiteral ).!
    def DecimalLiteral[$: P] = P( Digit.rep(1).! )
    def HexadecimalLiteral[$: P] = P( "0x" ~ HexDigit.rep(1) ).!
    def FloatLiteral[$: P] = P( CharIn("\\-\\+").? ~ DecimalLiteral ~ "." ~ DecimalLiteral ~ (CharIn("eE") ~ CharIn("\\-\\+").? ~ DecimalLiteral).?).! // substituted [0-9]* for [0-9]+
    val excludedCharacters: Set[Char] = Set('\"', '\n', '\f', '\u000B', '\r')
    def notExcluded[$: P] = P( CharPred(char => !excludedCharacters.contains(char)) )
    def StringLiteral[$: P] = P("\"" ~ notExcluded.rep ~ "\"").!


    /////////////////
    // IDENTIFIERS // 
    /////////////////

    // bare-id ::= (letter|[_]) (letter|digit|[_$.])*
    // bare-id-list ::= bare-id (`,` bare-id)*
    // value-id ::= `%` suffix-id
    // alias-name :: = bare-id
    // suffix-id ::= (digit+ | ((letter|id-punct) (letter|id-punct|digit)*))

    // symbol-ref-id ::= `@` (suffix-id | string-literal) (`::` symbol-ref-id)?
    // value-id-list ::= value-id (`,` value-id)*

    // // Uses of value, e.g. in an operand list to an operation.
    // value-use ::= value-id (`#` decimal-literal)?
    // value-use-list ::= value-use (`,` value-use)*


    def main(args: Array[String]): Unit = {
    
        val indent = "  "

        println("COMMON SYNTAX")
        // Digits
        val Parsed.Failure(_, _, _) = parse("a", Digit(_))
        val Parsed.Failure(_, _, _) = parse(" $ ! £ 4 1 ", Digit(_))
        val Parsed.Success("7", _) = parse("7", Digit(_))
        println(indent + "Digits - OK")
        // HexDigits
        val Parsed.Success("5", _) = parse("5", HexDigit(_))
        val Parsed.Success("f", _) = parse("f", HexDigit(_))
        val Parsed.Success("E", _) = parse("E", HexDigit(_))
        val Parsed.Failure(_, _, _) = parse("G", HexDigit(_))
        val Parsed.Failure(_, _, _) = parse("g", HexDigit(_))
        val Parsed.Success("4", _) = parse("41", HexDigit(_))
        println(indent + "HexDigits - OK")
        // Letters
        val Parsed.Success("a", _) = parse("a", Letter(_))
        val Parsed.Success("G", _) = parse("G", Letter(_))
        val Parsed.Failure(_, _, _) = parse("4", Letter(_))
        println(indent + "Letters - OK")
        // IdPuncts
        val Parsed.Success("$", _) = parse("$", IdPunct(_))
        val Parsed.Success(".", _) = parse(".", IdPunct(_))
        val Parsed.Success("_", _) = parse("_", IdPunct(_))
        val Parsed.Success("-", _) = parse("-", IdPunct(_))
        val Parsed.Failure(_, _, _) = parse("%", IdPunct(_))
        val Parsed.Failure(_, _, _) = parse("£", IdPunct(_))
        val Parsed.Failure(_, _, _) = parse("dfd", IdPunct(_))
        val Parsed.Failure(_, _, _) = parse("0", IdPunct(_))
        println(indent + "IdPunct - OK")
        // IntegerLiterals
        val Parsed.Success("123456789", _) = parse("123456789", IntegerLiteral(_))
        val Parsed.Success("1231", _) = parse("1231f", IntegerLiteral(_))
        val Parsed.Failure(_, _, _) = parse("f1231", IntegerLiteral(_))
        val Parsed.Success("0x0011ffff", _) = parse("0x0011ffff", IntegerLiteral(_))
        val Parsed.Success("0x0011", _) = parse("0x0011gggg", IntegerLiteral(_))
        val Parsed.Success("1", _) = parse("1xds%", IntegerLiteral(_))
        val Parsed.Success("0", _) = parse("0xgg", IntegerLiteral(_))
        println(indent + "IntegerLiteral - OK")
        // DecimalLiteral
        val Parsed.Success("123456789", _) = parse("123456789", DecimalLiteral(_))
        val Parsed.Success("1231", _) = parse("1231f", DecimalLiteral(_))
        val Parsed.Failure(_, _, _) = parse("f1231", DecimalLiteral(_))
        println(indent + "DecimalLiteral - OK")
        // HexadecimalLiteral
        val Parsed.Success("0x0011ffff", _) = parse("0x0011ffff", HexadecimalLiteral(_))
        val Parsed.Success("0x0011", _) = parse("0x0011gggg", HexadecimalLiteral(_))
        val Parsed.Failure(_, _, _) = parse("1xds%", HexadecimalLiteral(_))
        val Parsed.Failure(_, _, _) = parse("0xgg", HexadecimalLiteral(_))
        println(indent + "HexadecimalLiteral - OK")
        // FloatLiteral
        val Parsed.Success("1.0", _) = parse("1.0", FloatLiteral(_))
        val Parsed.Success("1.01242", _) = parse("1.01242", FloatLiteral(_))
        val Parsed.Success("993.013131", _) = parse("993.013131", FloatLiteral(_))
        val Parsed.Success("1.0e10", _) = parse("1.0e10", FloatLiteral(_))
        val Parsed.Success("1.0E10", _) = parse("1.0E10", FloatLiteral(_))
        val Parsed.Failure(_, _, _) = parse("1.", FloatLiteral(_))
        val Parsed.Success("1.0E10", _) = parse("1.0E10", FloatLiteral(_))
        val Parsed.Success("1.0E10", _) = parse("1.0E10", FloatLiteral(_))
        println(indent + "FloatLiteral - OK")
        // StringLiteral
        val Parsed.Success("\"hello\"", _) = parse("\"hello\"", StringLiteral(_))
        println(indent + "StringLiteral - OK")


    }
}

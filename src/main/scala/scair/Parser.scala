package scair

import fastparse._, MultiLineWhitespace._
import scala.collection.mutable
import Main._

object Parser {

    //////////////////////
    // COMMON FUNCTIONS //
    //////////////////////

    val typeAttributeMap: mutable.Map[String, Attribute] = mutable.Map.empty[String, Attribute]
    val valueMap: mutable.Map[String, Value] = mutable.Map.empty[String, Value]

    // not used just yet I am not sure as it is whether it will be useful
    def flattenList(commonList: (Any, Seq[Any])): Seq[Any] = {
        return commonList._1 +: commonList._2 
    }


    ///////////////////
    // COMMON SYNTAX // 
    ///////////////////

    // [x] digit     ::= [0-9]
    // [x] hex_digit ::= [0-9a-fA-F]
    // [x] letter    ::= [a-zA-Z]
    // [x] id-punct  ::= [$._-]

    // [x] integer-literal ::= decimal-literal | hexadecimal-literal
    // [x] decimal-literal ::= digit+
    // [x] hexadecimal-literal ::= `0x` hex_digit+
    // [x] float-literal ::= [-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?
    // [x] string-literal  ::= `"` [^"\n\f\v\r]* `"`

    def Digit[$: P] = P( CharIn("0-9").! )
    def HexDigit[$: P] = P( CharIn("0-9a-fA-F").! )
    def Letter[$: P] = P( CharIn("a-zA-Z").! )
    def IdPunct[$: P] = P( CharIn("$._\\-").! )

    def IntegerLiteral[$: P] = P( HexadecimalLiteral | DecimalLiteral ).!
    def DecimalLiteral[$: P] = P( Digit.rep(1).! )
    def HexadecimalLiteral[$: P] = P( "0x" ~~ HexDigit.rep(1) ).!
    def FloatLiteral[$: P] = P( CharIn("\\-\\+").? ~~ DecimalLiteral ~~ "." ~~ DecimalLiteral ~~ (CharIn("eE") ~~ CharIn("\\-\\+").? ~~ DecimalLiteral).?).! // substituted [0-9]* for [0-9]+
    val excludedCharacters: Set[Char] = Set('\"', '\n', '\f', '\u000B', '\r')  // \u000B represents \v escape character 
    def notExcluded[$: P] = P( CharPred(char => !excludedCharacters.contains(char)) )
    def StringLiteral[$: P] = P("\"" ~~ notExcluded.rep.! ~~ "\"")


    /////////////////
    // IDENTIFIERS // 
    /////////////////

    // [x] bare-id ::= (letter|[_]) (letter|digit|[_$.])*
    // [ ] bare-id-list ::= bare-id (`,` bare-id)*
    // [x] value-id ::= `%` suffix-id
    // [x] alias-name :: = bare-id
    // [x] suffix-id ::= (digit+ | ((letter|id-punct) (letter|id-punct|digit)*))
    
    // [ ] symbol-ref-id ::= `@` (suffix-id | string-literal) (`::` symbol-ref-id)?
    // [ ] value-id-list ::= value-id (`,` value-id)*
    
    // // Uses of value, e.g. in an operand list to an operation.
    // [x] value-use ::= value-id (`#` decimal-literal)?
    // [x] value-use-list ::= value-use (`,` value-use)*

    def flattenValueUseList(valueUseList: (String, Option[String], Seq[(String, Option[String])])): Seq[(String, Option[String])] = {
        return (valueUseList._1, valueUseList._2) +: valueUseList._3 
    }
    
    def BareId[$: P] = P( ( Letter | "_" ) ~~ ( Letter | Digit | CharIn("_$.") ).rep ).!
    def ValueId[$: P] = P( "%" ~~ SuffixId )
    def AliasName[$: P] = P( BareId )
    def SuffixId[$: P] = P( DecimalLiteral | ( Letter | IdPunct ) ~~ ( Letter | IdPunct | Digit ).rep ).!

    def ValueUse[$: P] = P( ValueId ~ ( "#" ~ DecimalLiteral ).? )
    def ValueUseList[$: P] = P( ValueUse ~ ( "," ~ ValueUse ).rep ).map(flattenValueUseList)


    ////////////////
    // OPERATIONS // 
    ////////////////
        
    // [x] operation             ::= op-result-list? (generic-operation | custom-operation)
    //                         trailing-location?
    // [x] generic-operation     ::= string-literal `(` value-use-list? `)`  successor-list?
    //                         dictionary-properties? region-list? dictionary-attribute?
    //                         `:` function-type
    // [ ] custom-operation      ::= bare-id custom-operation-format
    // [x] op-result-list        ::= op-result (`,` op-result)* `=`
    // [x] op-result             ::= value-id (`:` integer-literal)?
    // [ ] successor-list        ::= `[` successor (`,` successor)* `]`
    // [ ] successor             ::= caret-id (`:` block-arg-list)?
    // [ ] dictionary-properties ::= `<` dictionary-attribute `>`
    // [ ] region-list           ::= `(` region (`,` region)* `)`
    // [ ] dictionary-attribute  ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
    // [x] trailing-location     ::= `loc` `(` location `)`

    //
    // operation: (name, parameters, type of function)
    // 
    def generateOperation(operation: (String, Option[Seq[(String, Option[String])]], (Any, Any))): (String, Option[Seq[(String, Option[String])]], (Any, Any)) = {
        println(operation._1)
        //return ("", Option(""), "")
        val (operandsTypes, resultTypes) = operation._3

        println(operandsTypes.getClass())

        return operation
    }

    def joinResultList(results: (String, Option[String], Seq[(String, Option[String])])): Seq[String] = {
        
        val ssaValueIds = for {
            n <- results._3
        } yield n._1

        return results._1 +: ssaValueIds
    }

    def OperationPat[$: P] = P( OpResultList.? ~ GenericOperation ~ TrailingLocation.!.?) // shortened definition TODO: finish...
    def GenericOperation[$: P] = P( StringLiteral ~ "(" ~ ValueUseList.? ~ ")" ~ ":" ~ FunctionType ).map(generateOperation) // shortened definition TODO: finish...
    def OpResultList[$: P] = P( OpResult ~ ( "," ~ OpResult ).rep ~ "=" ).map(joinResultList)
    def OpResult[$: P] = P( ValueId ~ ( ":" ~ IntegerLiteral ).? )
    def TrailingLocation[$: P] = P( "loc" ~ "(" ~ "unknown" ~ ")" ) // definition taken from xdsl attribute_parser.py v-0.19.0 line 1106
    

    //////////////////////////
    // TOP LEVEL PRODUCTION //
    //////////////////////////

    // toplevel := (operation | attribute-alias-def | type-alias-def)*

    def TopLevel[$: P] = P( OperationPat.!.rep ~ End ) // shortened definition TODO: finish...


    ///////////
    // Types //
    ///////////

    // [x] type ::= type-alias | dialect-type | builtin-type
    
    // [x] type-list-no-parens ::=  type (`,` type)*
    // [x] type-list-parens ::= `(` `)` | `(` type-list-no-parens `)`
    
    // // This is a common way to refer to a value with a specified type.
    // [ ] ssa-use-and-type ::= ssa-use `:` type
    // [ ] ssa-use ::= value-use
    
    // // Non-empty list of names and types.
    // [ ] ssa-use-and-type-list ::= ssa-use-and-type (`,` ssa-use-and-type)*
    
    // [ ] function-type ::= (type | type-list-parens) `->` (type | type-list-parens)
    
    // // Type aliases
    // [x] type-alias-def ::= `!` alias-name `=` type
    // [x] type-alias ::= `!` alias-name
    
    def joinTypeList(types: (String, Seq[String])): Seq[String] = {
        return types._1 +: types._2 
    }

    def mapAttribute(typeName: String): String = {

        if (!typeAttributeMap.contains(typeName)) {
            typeAttributeMap(typeName) = new Attribute(name = typeName)
        }
        return typeName
    }

    def I32[$: P] = P("i32".!)
    def I64[$: P] = P("i64".!)
    def BuiltIn[$: P] = P(I32 | I64) // temporary BuiltIn

    def Type[$: P] = P( TypeAlias | BuiltIn ).map(mapAttribute) // shortened definition TODO: finish...

    def TypeListNoParens[$: P] = P( Type ~ ( "," ~ Type ).rep ).map(joinTypeList)
    def TypeListParens[$: P] = P( "(" ~ ")" | "(" ~ TypeListNoParens ~ ")" )

    def FunctionType[$: P] = P( ( Type | TypeListParens ) ~ "->" ~ ( Type | TypeListParens ) )

    def TypeAliasDef[$: P] = P( "!" ~~ AliasName ~ "=" ~ Type )
    def TypeAlias[$: P] = P( "!" ~~ AliasName )

    
    /////////////
    // TESTING //
    /////////////

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
        val Parsed.Success("hello", _) = parse("\"hello\"", StringLiteral(_))
        val Parsed.Success("hello", _) = parse("\"hello\"", StringLiteral(_))
        println(indent + "StringLiteral - OK")


        println("IDENTIFIERS")
        // ValueId
        val Parsed.Success("hello", _) = parse("%hello", ValueId(_))
        val Parsed.Success("Ater", _) = parse("%Ater", ValueId(_))
        val Parsed.Success("312321", _) = parse("%312321", ValueId(_))
        val Parsed.Success("Ater", _) = parse("%Ater", ValueId(_))
        val Parsed.Success("$$$$$", _) = parse("%$$$$$", ValueId(_))
        val Parsed.Success("_-_-_", _) = parse("%_-_-_", ValueId(_))
        val Parsed.Success("3", _) = parse("%3asada", ValueId(_))
        val Parsed.Failure(_, _, _) = parse("% hello", ValueId(_))
        println(indent + "ValueId - OK")
        //


        println("OPERATIONS")
        // OpResultList
        println(parse("%0, %1, %2 =", OpResultList(_)))
        // val Parsed.Success(Seq("%0", "%1", "%2"), _) = parse("%0, %1, %2 =", OpResultList(_))
        // val Parsed.Success(Seq("%0", "%1", "%2"), _) = parse("%0   ,    %1   ,   %2  =   ", OpResultList(_))
        // val Parsed.Success(Seq("%0", "%1", "%2"), _) = parse("%0,%1,%2=", OpResultList(_))
        println(indent + "OpResultList - OK")

        println("TOP LEVEL PRODUCTION")

        println("----In-Progress----")
        println(parse("%0, %1, %2 = \"test.op\"() : () -> (i32, i64, i32)", OperationPat(_))) 
        println(parse("\"test.op\"(%1, %0) : (i64, i32) -> ()", OperationPat(_))) 
        println(typeAttributeMap)
        
    }
}

package scair

import fastparse._, MultiLineWhitespace._
import scala.collection.mutable
import scala.util.{Try,Success,Failure}
import Main._

object Parser {

    //////////////////////
    // COMMON FUNCTIONS //
    //////////////////////

    // val typeAttributeMap: mutable.Map[String, Attribute] = mutable.Map.empty[String, Attribute]
    // var valueMap: mutable.Map[String, Value] = mutable.Map.empty[String, Value]

    // not used just yet I am not sure as it is whether it will be useful
    def flattenList(commonList: (Any, Seq[Any])): Seq[Any] = {
        return commonList._1 +: commonList._2 
    }

    def parser(valueMap: mutable.Map[String, Value] =  mutable.Map.empty[String, Value], text: String): fastparse.Parsed[Seq[Operation]] = {

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

        def decimalToInt(literal: String): Int = {
            return literal.toInt
        }

        def hexToInt(hex: String): Int = {
            return Integer.parseInt(hex, 16)
        }

        def Digit[$: P] = P( CharIn("0-9").! )
        def HexDigit[$: P] = P( CharIn("0-9a-fA-F").! )
        def Letter[$: P] = P( CharIn("a-zA-Z").! )
        def IdPunct[$: P] = P( CharIn("$._\\-").! )

        def IntegerLiteral[$: P] = P( HexadecimalLiteral | DecimalLiteral )
        def DecimalLiteral[$: P] = P( Digit.rep(1).! ).map(decimalToInt)
        def HexadecimalLiteral[$: P] = P( "0x" ~~ HexDigit.rep(1).! ).map(hexToInt)
        def FloatLiteral[$: P] = P( CharIn("\\-\\+").? ~~ DecimalLiteral ~~ "." ~~ DecimalLiteral ~~ (CharIn("eE") ~~ CharIn("\\-\\+").? ~~ DecimalLiteral).?).! // substituted [0-9]* with [0-9]+
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

        def flattenValueUseList(valueUseList: (String, Seq[String])): Seq[String] = {
            return valueUseList._1 +: valueUseList._2 
        }

        def simplifyValueName(valueUse: (String, Option[Int])): String = valueUse match {
            case (name, Some(number)) => s"$name#$number"
            case (name, None) => name
        }
        
        def BareId[$: P] = P( ( Letter | "_" ) ~~ ( Letter | Digit | CharIn("_$.") ).rep ).!
        def ValueId[$: P] = P( "%" ~~ SuffixId )
        def AliasName[$: P] = P( BareId )
        def SuffixId[$: P] = P( DecimalLiteral | ( Letter | IdPunct ) ~~ ( Letter | IdPunct | Digit ).rep ).!

        def ValueUse[$: P] = P( ValueId ~ ( "#" ~ DecimalLiteral ).? ).map(simplifyValueName)
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
        // operation: (results, (name, operands, type of function))
        // 
        def generateOperation(operation: (Option[Seq[String]], (String, Option[Seq[String]], (Seq[Attribute], Seq[Attribute])))): Operation = {
            
            val results: Seq[String] = operation._1 match { case None => Seq[String]() case Some(x) => x}
            val opName = operation._2._1
            val operands: Seq[String] = operation._2._2 match { case None => Seq[String]() case Some(x) => x}
            val resultsTypes = operation._2._3._2
            val operandsTypes = operation._2._3._1

            if ( results.length != resultsTypes.length ) {
                throw new Exception("E")
            }

            if ( operands.length != operandsTypes.length ) {
                throw new Exception("E")
            }
            
            // val resultss: Seq[String, Value] = for { n <- (0 to results.length) } yield (results(n), new Value(typ = resultsTypes(n)))
            // val operandss: Seq[String, Value] = for { n <- (0 to operands.length) } yield (operands(n), new Value(typ = operandsTypes(n)))

            var resultss: Seq[Value] = Seq()
            var operandss: Seq[Value] = Seq()

            for (i <- (0 to results.length - 1)) {
                val name = results(i)
                val value = new Value(typ = resultsTypes(i))

                // SSA value cannot be defined twice
                if (valueMap.contains(name)) {
                    throw new Exception(s"SSA value cannot be defined twice %$name")
                }

                valueMap(name) = value

                resultss = resultss :+ value
            }

            for (i <- (0 to operands.length - 1)) {
                val name = operands(i)
                // val value = new Value(typ = operandsTypes(i))

                // used but not defined 
                if (!valueMap.contains(name)) {
                    // valueMap(name) = value
                    throw new Exception(s"SSA value used but not defined %$name")
                }
                if (valueMap(name).typ != operandsTypes(i)) {
                    throw new Exception(s"$name use with type ${operandsTypes(i)} but defined with type ${valueMap(name).typ}")
                }
                operandss = operandss :+ valueMap(name)
            }

            return new Operation(name = opName, operands = operandss, results = resultss)
        }

        def joinResultList(results: (Seq[String], Seq[Seq[String]])): Seq[String] = {
            return results._1 ++ results._2.flatten
        }

        def sequenceValues(value: (String, Option[Int])): Seq[String] = value match {
            case (name, Some(totalNo)) => (0 to totalNo).map(no => s"$name#$no")
            case (name, None) => Seq(name)
        }

        def OperationPat[$: P] = P( OpResultList.? ~ GenericOperation ~ TrailingLocation.?).map(generateOperation) // shortened definition TODO: finish...
        def GenericOperation[$: P] = P( StringLiteral ~ "(" ~ ValueUseList.? ~ ")" ~ ":" ~ FunctionType ) // shortened definition TODO: finish...
        def OpResultList[$: P] = P( OpResult ~ ( "," ~ OpResult ).rep ~ "=" ).map(joinResultList)
        def OpResult[$: P] = P( ValueId ~ ( ":" ~ IntegerLiteral ).? ).map(sequenceValues)
        def TrailingLocation[$: P] = P( "loc" ~ "(" ~ "unknown" ~ ")" ) // definition taken from xdsl attribute_parser.py v-0.19.0 line 1106
        

        //////////////////////////
        // TOP LEVEL PRODUCTION //
        //////////////////////////

        // toplevel := (operation | attribute-alias-def | type-alias-def)*

        def TopLevel[$: P] = P( OperationPat.rep ~ End ) // shortened definition TODO: finish...


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
        
        def joinTypeList(types: (Seq[Attribute], Seq[Seq[Attribute]])): Seq[Attribute] = {
            return types._1 ++ types._2.flatten
        }

        def mapAttribute(typeName: String): Seq[Attribute] = {

            // if (!typeAttributeMap.contains(typeName)) {
            //     typeAttributeMap(typeName) = new Attribute(name = typeName)
            // }
            return Seq(new Attribute(name = typeName))
        }

        // def sequent(value: Any): Seq[Attribute] = value match {
        //     case x: Seq[Attribute] => x
        //     case y: Any => Seq()
        // }

        def I32[$: P] = P("i32".!)
        def I64[$: P] = P("i64".!)
        def BuiltIn[$: P] = P(I32 | I64) // temporary BuiltIn

        def Type[$: P] = P( TypeAlias | BuiltIn ).map(mapAttribute) // shortened definition TODO: finish...

        def TypeListNoParens[$: P] = P( Type ~ ( "," ~ Type ).rep ).map(joinTypeList)
        def TypeListParens[$: P] = P( ClosedParens | "(" ~ TypeListNoParens ~ ")" )
        def ClosedParens[$: P] = P( "(" ~ ")" ).map(_ => Seq[Attribute]())

        def FunctionType[$: P] = P( ( Type | TypeListParens ) ~ "->" ~ ( Type | TypeListParens ) )

        def TypeAliasDef[$: P] = P( "!" ~~ AliasName ~ "=" ~ Type )
        def TypeAlias[$: P] = P( "!" ~~ AliasName )

        return parse(text, TopLevel(_))
    }
    
    /////////////
    // TESTING //
    /////////////

    def main(args: Array[String]): Unit = {
    
        val indent = "  "
        
        //valueMap = mutable.Map.empty[String, Value]

        // println("COMMON SYNTAX")
        // // Digits
        // val Parsed.Failure(_, _, _) = parse("a", Digit(_))
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

        println("TOP LEVEL PRODUCTION")

        val valueMap1 = mutable.Map.empty[String, Value]
        val valueMap2 = mutable.Map.empty[String, Value]

        println("----In-Progress----")
        println(parser(text = "%0, %1, %2 = \"test.op\"() : () -> (i32, i64, i32)"))
        println(parser(valueMap2, "%0, %1, %2 = \"test.op\"() : () -> (i32, i64, i32)\n" +
         "\"test.op\"(%1, %0) : (i64, i32) -> ()"))
        // println(parse("%0, %1, %2 = \"test.op\"() : () -> (i32, i64, i32)\n" +
        // "\"test.op\"(%1, %0) : (i64, i32) -> ()", TopLevel(_))) 
        // valueMap = mutable.Map.empty[String, Value]
        // println(parse("\"test.op\"(%1, %0) : (i64, i32) -> ()", TopLevel(_))) 
        
    }
}

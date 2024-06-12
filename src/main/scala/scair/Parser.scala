package scair

import fastparse._, MultiLineWhitespace._
import scala.collection.mutable
import scala.util.{Try, Success, Failure}
import Main._

object Parser {

  //////////////////////
  // COMMON FUNCTIONS //
  //////////////////////

  var valueMap: mutable.Map[String, Value] = mutable.Map.empty[String, Value]

  // context is kept in a class, and each one automatically swaps out the valueMap based on its local value map
  class Context(var localValueMap: mutable.Map[String, Value]) {

    // we first slot in the local context into the Parser object, then slot it back out once parsing is done
    def parseThis(text: String): fastparse.Parsed[Seq[Operation]] = {
      valueMap = localValueMap
      val result = parse(text, TopLevel(_))
      localValueMap = valueMap
      return result
    }

    // child starts off from the parents context
    def createChild(): Context = {
      return new Context(localValueMap = localValueMap)
    }

    def getLocalValueMap(): mutable.Map[String, Value] = localValueMap

    def testParse[A](
        text: String,
        parser: fastparse.P[_] => fastparse.P[A]
    ): fastparse.Parsed[A] = {
      valueMap = mutable.Map.empty[String, Value]
      return parse(text, parser)
    }
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

  def Digit[$: P] = P(CharIn("0-9").!)
  def HexDigit[$: P] = P(CharIn("0-9a-fA-F").!)
  def Letter[$: P] = P(CharIn("a-zA-Z").!)
  def IdPunct[$: P] = P(CharIn("$._\\-").!)

  def IntegerLiteral[$: P] = P(HexadecimalLiteral | DecimalLiteral)
  def DecimalLiteral[$: P] =
    P(Digit.rep(1).!).map((literal: String) => literal.toInt)
  def HexadecimalLiteral[$: P] =
    P("0x" ~~ HexDigit.rep(1).!).map((hex: String) => Integer.parseInt(hex, 16))
  def FloatLiteral[$: P] = P(
    CharIn("\\-\\+").? ~~ DecimalLiteral ~~ "." ~~ DecimalLiteral ~~ (CharIn(
      "eE"
    ) ~~ CharIn("\\-\\+").? ~~ DecimalLiteral).?
  ).! // substituted [0-9]* with [0-9]+
  val excludedCharacters: Set[Char] = Set('\"', '\n', '\f', '\u000B',
    '\r') // \u000B represents \v escape character
  def notExcluded[$: P] = P(
    CharPred(char => !excludedCharacters.contains(char))
  )
  def StringLiteral[$: P] = P("\"" ~~ notExcluded.rep.! ~~ "\"")

  //////////////////////////
  // TOP LEVEL PRODUCTION //
  //////////////////////////

  // [x] toplevel := (operation | attribute-alias-def | type-alias-def)*

  def TopLevel[$: P] = P(
    OperationPat.rep ~ End
  ) // shortened definition TODO: finish...

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

  def simplifyValueName(valueUse: (String, Option[Int])): String =
    valueUse match {
      case (name, Some(number)) => s"$name#$number"
      case (name, None)         => name
    }

  def BareId[$: P] = P((Letter | "_") ~~ (Letter | Digit | CharIn("_$.")).rep).!
  def ValueId[$: P] = P("%" ~~ SuffixId)
  def AliasName[$: P] = P(BareId)
  def SuffixId[$: P] = P(
    DecimalLiteral | (Letter | IdPunct) ~~ (Letter | IdPunct | Digit).rep
  ).!

  def ValueUse[$: P] =
    P(ValueId ~ ("#" ~ DecimalLiteral).?).map(simplifyValueName)
  def ValueUseList[$: P] = P(ValueUse ~ ("," ~ ValueUse).rep).map(
    (valueUseList: (String, Seq[String])) => valueUseList._1 +: valueUseList._2
  )

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

  //                                     results           name        operands            op types       res types
  def generateOperation(
      operation: (
          Option[Seq[String]],
          (String, Option[Seq[String]], (Seq[Attribute], Seq[Attribute]))
      )
  ): Operation = {

    val results: Seq[String] = operation._1 match {
      case None    => Seq[String]()
      case Some(x) => x
    }
    val opName = operation._2._1
    val operands: Seq[String] = operation._2._2 match {
      case None    => Seq[String]()
      case Some(x) => x
    }
    val resultsTypes = operation._2._3._2
    val operandsTypes = operation._2._3._1

    if (results.length != resultsTypes.length) {
      throw new Exception("E")
    }

    if (operands.length != operandsTypes.length) {
      throw new Exception("E")
    }

    val resultss: Seq[Value] = for {
      n <- (0 to results.length - 1)
    } yield valueMap.contains(results(n)) match {
      case true =>
        throw new Exception(s"SSA Value cannot be defined twice %${results(n)}")
      case false =>
        val value = new Value(typ = resultsTypes(n))
        valueMap(results(n)) = value
        value
    }

    val operandss: Seq[Value] = for {
      n <- { 0 to operands.length - 1 }
    } yield !valueMap.contains(operands(n)) match {
      case true =>
        throw new Exception(s"SSA value used but not defined %${operands(n)}")
      case false =>
        val name = operands(n)
        valueMap(name).typ != operandsTypes(n) match {
          case true =>
            throw new Exception(
              s"$name use with type ${operandsTypes(n)} but defined with type ${valueMap(name).typ}"
            )
          case false =>
            valueMap(name)
        }
    }

    return new Operation(
      name = opName,
      operands = operandss,
      results = resultss
    )
  }

  def sequenceValues(value: (String, Option[Int])): Seq[String] = value match {
    case (name, Some(totalNo)) => (0 to totalNo).map(no => s"$name#$no")
    case (name, None)          => Seq(name)
  }

  def OperationPat[$: P] = P(
    OpResultList.? ~ GenericOperation ~ TrailingLocation.?
  ).map(generateOperation) // shortened definition TODO: finish...
  def GenericOperation[$: P] = P(
    StringLiteral ~ "(" ~ ValueUseList.? ~ ")" ~ ":" ~ FunctionType
  ) // shortened definition TODO: finish...
  def OpResultList[$: P] = P(OpResult ~ ("," ~ OpResult).rep ~ "=").map(
    (results: (Seq[String], Seq[Seq[String]])) =>
      results._1 ++ results._2.flatten
  )
  def OpResult[$: P] = P(ValueId ~ (":" ~ IntegerLiteral).?).map(sequenceValues)
  def TrailingLocation[$: P] = P(
    "loc" ~ "(" ~ "unknown" ~ ")"
  ) // definition taken from xdsl attribute_parser.py v-0.19.0 line 1106

  ////////////
  // BLOCKS //
  ////////////

  // [ ] - block           ::= block-label operation+
  // [ ] - block-label     ::= block-id block-arg-list? `:`
  // [ ] - block-id        ::= caret-id
  // [ ] - caret-id        ::= `^` suffix-id
  // [ ] - value-id-and-type ::= value-id `:` type

  // // Non-empty list of names and types.
  // [ ] - value-id-and-type-list ::= value-id-and-type (`,` value-id-and-type)*

  // [ ] - block-arg-list ::= `(` value-id-and-type-list? `)`

  /////////////
  // REGIONS //
  /////////////

  // [ ] - region        ::= `{` entry-block? block* `}`
  // [ ] - entry-block   ::= operation+

  ///////////
  // TYPES //
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

  def I32[$: P] = P("i32".!)
  def I64[$: P] = P("i64".!)
  def BuiltIn[$: P] = P(I32 | I64) // temporary BuiltIn

  def Type[$: P] = P(TypeAlias | BuiltIn).map((typeName: String) =>
    Seq(new Attribute(name = typeName))
  ) // shortened definition TODO: finish...

  def TypeListNoParens[$: P] = P(Type ~ ("," ~ Type).rep).map(
    (types: (Seq[Attribute], Seq[Seq[Attribute]])) =>
      types._1 ++ types._2.flatten
  )
  def TypeListParens[$: P] = P(ClosedParens | "(" ~ TypeListNoParens ~ ")")
  def ClosedParens[$: P] = P("(" ~ ")").map(_ => Seq[Attribute]())

  def FunctionType[$: P] = P(
    (Type | TypeListParens) ~ "->" ~ (Type | TypeListParens)
  )

  def TypeAliasDef[$: P] = P("!" ~~ AliasName ~ "=" ~ Type)
  def TypeAlias[$: P] = P("!" ~~ AliasName)

  ////////////////
  // ATTRIBUTES //
  ////////////////

  // [ ] - attribute-entry ::= (bare-id | string-literal) `=` attribute-value
  // [ ] - attribute-value ::= attribute-alias | dialect-attribute | builtin-attribute

  // // Attribute Value Aliases
  // [ ] - attribute-alias-def ::= `#` alias-name `=` attribute-value
  // [ ] - attribute-alias ::= `#` alias-name

  ///////////////////
  // DIALECT TYPES //
  ///////////////////

  // [ ] - dialect-namespace      ::= bare-id

  // [ ] - dialect-type           ::= `!` (opaque-dialect-type | pretty-dialect-type)
  // [ ] - opaque-dialect-type    ::= dialect-namespace dialect-type-body
  // [ ] - pretty-dialect-type    ::= dialect-namespace `.` pretty-dialect-type-lead-ident dialect-type-body?
  // [ ] - pretty-dialect-type-lead-ident ::= `[A-Za-z][A-Za-z0-9._]*`

  // [ ] - dialect-type-body      ::= `<` dialect-type-contents+ `>`
  // [ ] - dialect-type-contents  ::= dialect-type-body
  //                             | `(` dialect-type-contents+ `)`
  //                             | `[` dialect-type-contents+ `]`
  //                             | `{` dialect-type-contents+ `}`
  //                             | [^\[<({\]>)}\0]+

  //////////////////////////////
  // DIALECT ATTRIBUTE VALUES //
  //////////////////////////////

  // [ ] - dialect-namespace ::= bare-id

  // [ ] - dialect-attribute ::= `#` (opaque-dialect-attribute | pretty-dialect-attribute)
  // [ ] - opaque-dialect-attribute ::= dialect-namespace dialect-attribute-body
  // [ ] - pretty-dialect-attribute ::= dialect-namespace `.` pretty-dialect-attribute-lead-ident dialect-attribute-body?
  // [ ] - pretty-dialect-attribute-lead-ident ::= `[A-Za-z][A-Za-z0-9._]*`

  // [ ] - dialect-attribute-body ::= `<` dialect-attribute-contents+ `>`
  // [ ] - dialect-attribute-contents ::= dialect-attribute-body
  //                             | `(` dialect-attribute-contents+ `)`
  //                             | `[` dialect-attribute-contents+ `]`
  //                             | `{` dialect-attribute-contents+ `}`
  //                             | [^\[<({\]>)}\0]+

  //////////////////////////////////
  // FUNCTIONAL TESTING ON THE GO //
  //////////////////////////////////

  def main(args: Array[String]): Unit = {

    println("---- IN PROGRESS ----")

    val context1 = new Context(localValueMap = mutable.Map.empty[String, Value])

    println(
      context1.testParse(
        text = "%0, %1, %2 = \"test.op\"() : () -> (i32, i64, i32)",
        parser = TopLevel(_)
      )
    )
    println(
      context1.testParse(
        text = "%0, %1, %2 = \"test.op\"() : () -> (i32, i64, i32)\n" +
          "\"test.op\"(%1, %0) : (i64, i32) -> ()",
        parser = TopLevel(_)
      )
    )

    println("---------------------")
  }
}

/////////////////////
// VERSION CONTROL //
/////////////////////

/*

def generateOperation

    var resultss: Seq[Value] = Seq()
    var operandss: Seq[Value] = Seq()

    for (i <- (0 to results.length - 1)) {
        val name = results(i)
        val value = new Value(typ = resultsTypes(i))

        // SSA value cannot be defined twice
        if (valueMap.contains(name)) {
            throw new Exception(s"SSA Value cannot be defined twice %$name")
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


    ///////////
    // Types //
    ///////////

    def joinTypeList(types: (Seq[Attribute], Seq[Seq[Attribute]])): Seq[Attribute] = {
        return types._1 ++ types._2.flatten
    }

    def mapAttribute(typeName: String): Seq[Attribute] = {

        // if (!typeAttributeMap.contains(typeName)) {
        //     typeAttributeMap(typeName) = new Attribute(name = typeName)
        // }
        return Seq(new Attribute(name = typeName))
    }


 */

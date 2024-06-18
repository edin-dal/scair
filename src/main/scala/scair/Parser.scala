package scair

import fastparse._, MultiLineWhitespace._
import scala.collection.mutable
import scala.util.{Try, Success, Failure}
import IR._

object Parser {}

class Parser {

  // Custom function wrapper that allows to Escape out of a pattern
  // to carry out custom computation
  def E[_: P](action: => Unit) = {
    action
    Pass()
  }

  object Scope {

    def defineValues(
        valueIdAndTypeList: Seq[(String, Attribute)]
    )(implicit
        scope: Scope
    ): Seq[Value] = {
      for {
        (name, typ) <- valueIdAndTypeList
      } yield scope.valueMap.contains(name) match {
        case true =>
          // val x: fastparse.Parsed.Extra = fastparse.Parsed.Extra
          // Parsed.Failure("", 0, x)
          throw new Exception(s"SSA Value cannot be defined twice %${name}")
        case false =>
          val value = new Value(typ = typ)
          scope.valueMap(name) = value
          value
      }
    }

    def useValues(
        valueIdAndTypeList: Seq[(String, Attribute)]
    )(implicit
        scope: Scope
    ): Seq[Value] = {
      for {
        (name, typ) <- valueIdAndTypeList
      } yield !scope.valueMap.contains(name) match {
        case true =>
          throw new Exception(s"SSA value used but not defined %${name}")
        case false =>
          scope.valueMap(name).typ != typ match {
            case true =>
              throw new Exception(
                s"$name use with type ${typ} but defined with type ${scope.valueMap(name).typ}"
              )
            case false =>
              scope.valueMap(name)
          }
      }
    }

    def defineBlock(
        blockName: String,
        block: Block
    )(implicit
        scope: Scope
    ): Unit = scope.blockMap.contains(blockName) match {
      case true =>
        throw new Exception(
          s"Block cannot be defined twice within the same scope - ^${blockName}"
        )
      case false =>
        scope.blockMap(blockName) = block
    }

    def checkWaitlist()(implicit scope: Scope): Unit = {
      for ((operation, successors) <- scope.blockWaitlist) {
        val successorList: Seq[Block] = for {
          name <- successors
        } yield scope.blockMap
          .contains(
            name
          ) match {
          case false =>
            throw new Exception(s"Successor ${name} not defined within Scope")
          case true => scope.blockMap(name)
        }
        operation.successors = successorList
      }
    }
  }

  class Scope(
      var valueMap: mutable.Map[String, Value] =
        mutable.Map.empty[String, Value],
      var blockMap: mutable.Map[String, Block] =
        mutable.Map.empty[String, Block],
      var parentScope: Option[Scope] = None,
      var blockWaitlist: mutable.Map[Operation, Seq[String]] =
        mutable.Map.empty[Operation, Seq[String]]
  ) {

    // child starts off from the parents context
    def createChild(): Scope = {
      return new Scope(
        valueMap = valueMap,
        blockMap = blockMap,
        parentScope = Some(this)
      )
    }

    def switchWithChild(): Unit = {
      currentScope = createChild()
    }

    def switchWithParent(): Unit = parentScope match {
      case Some(x) =>
        Scope.checkWaitlist
        currentScope = x
      case None =>
        throw new Exception("No parent present - check your")
    }
  }

  implicit var currentScope: Scope = new Scope()

  //////////////////////
  // COMMON FUNCTIONS //
  //////////////////////

  def optionlessThis[A](option: Option[Seq[A]]): Seq[A] = option match {
    case None    => Seq[A]()
    case Some(x) => x
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

  val excludedCharacters: Set[Char] = Set('\"', '\n', '\f', '\u000B',
    '\r') // \u000B represents \v escape character

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
  // [x] successor-list        ::= `[` successor (`,` successor)* `]`
  // [x] successor             ::= caret-id (`:` block-arg-list)?
  // [ ] dictionary-properties ::= `<` dictionary-attribute `>`
  // [x] region-list           ::= `(` region (`,` region)* `)`
  // [ ] dictionary-attribute  ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
  // [x] trailing-location     ::= `loc` `(` location `)`

  //  results      name     operands   successors   regions   (op types      res types)
  def generateOperation(
      operation: (
          Seq[String],
          (
              String,
              Seq[String],
              Seq[String],
              Seq[Region],
              (Seq[Attribute], Seq[Attribute])
          )
      )
  ): Operation = {

    val results: Seq[String] = operation._1
    val opName = operation._2._1
    val operands: Seq[String] = operation._2._2
    val successors: Seq[String] = operation._2._3
    val regions: Seq[Region] = operation._2._4
    val resultsTypes = operation._2._5._2
    val operandsTypes = operation._2._5._1

    if (results.length != resultsTypes.length) {
      throw new Exception("E")
    }

    if (operands.length != operandsTypes.length) {
      throw new Exception("E")
    }

    val resultss: Seq[Value] = Scope.defineValues(results zip resultsTypes)

    val operandss: Seq[Value] = Scope.useValues(operands zip operandsTypes)

    val op = new Operation(
      name = opName,
      operands = operandss,
      successors = Seq(),
      results = resultss,
      regions = regions
    )

    currentScope.blockWaitlist += op -> successors

    return op
  }

  def sequenceValues(value: (String, Option[Int])): Seq[String] = value match {
    case (name, Some(totalNo)) => (0 to totalNo).map(no => s"$name#$no")
    case (name, None)          => Seq(name)
  }

  def OperationPat[$: P]: P[Operation] = P(
    OpResultList.?.map(optionlessThis) ~ GenericOperation ~ TrailingLocation.?
  ).map(generateOperation) // shortened definition TODO: finish...

  def GenericOperation[$: P] = P(
    StringLiteral ~ "(" ~ ValueUseList.?.map(
      optionlessThis
    ) ~ ")" ~ SuccessorList.?.map(optionlessThis) ~ RegionList.?.map(
      optionlessThis
    ) ~ ":" ~ FunctionType
  ) // shortened definition TODO: finish...

  def OpResultList[$: P] = P(OpResult ~ ("," ~ OpResult).rep ~ "=").map(
    (results: (Seq[String], Seq[Seq[String]])) =>
      results._1 ++ results._2.flatten
  )
  def OpResult[$: P] = P(ValueId ~ (":" ~ IntegerLiteral).?).map(sequenceValues)

  def SuccessorList[$: P] =
    P("[" ~ Successor ~ ("," ~ Successor).rep ~ "]").map(
      (successors: (String, Seq[String])) => successors._1 +: successors._2
    )
  def Successor[$: P] = P(CaretId) // possibly shortened version

  def RegionList[$: P] = P("(" ~ Region ~ ("," ~ Region).rep ~ ")")
    .map((x: (Region, Seq[Region])) => x._1 +: x._2)

  def TrailingLocation[$: P] = P(
    "loc" ~ "(" ~ "unknown" ~ ")"
  ) // definition taken from xdsl attribute_parser.py v-0.19.0 line 1106

  ////////////
  // BLOCKS //
  ////////////

  // [x] - block           ::= block-label operation+
  // [x] - block-label     ::= block-id block-arg-list? `:`
  // [x] - block-id        ::= caret-id
  // [x] - caret-id        ::= `^` suffix-id
  // [x] - value-id-and-type ::= value-id `:` type

  // // Non-empty list of names and types.
  // [x] - value-id-and-type-list ::= value-id-and-type (`,` value-id-and-type)*

  // [x] - block-arg-list ::= `(` value-id-and-type-list? `)`

  def createBlock(
      //            name    argument     operations
      uncutBlock: (String, Seq[Value], Seq[Operation])
  ): Block = {
    val newBlock = new Block(
      operations = uncutBlock._3,
      arguments = uncutBlock._2
    )

    Scope.defineBlock(uncutBlock._1, newBlock)
    return newBlock
  }

  def Block[$: P] = P(BlockLabel ~ OperationPat.rep(1)).map(createBlock)

  def BlockLabel[$: P] = P(
    BlockId ~ BlockArgList.?.map(optionlessThis)
      .map(Scope.defineValues) ~ ":"
  )

  def BlockId[$: P] = P(CaretId)

  def CaretId[$: P] = P("^" ~ SuffixId)

  def ValueIdAndType[$: P] = P(ValueId ~ ":" ~ Type)

  def ValueIdAndTypeList[$: P] =
    P(ValueIdAndType ~ ("," ~ ValueIdAndType).rep).map(
      (idAndTypes: (String, Attribute, Seq[(String, Attribute)])) =>
        (idAndTypes._1, idAndTypes._2) +: idAndTypes._3
    )

  def BlockArgList[$: P] =
    P("(" ~ ValueIdAndTypeList.? ~ ")").map(optionlessThis)

  /////////////
  // REGIONS //
  /////////////

  // [x] - region        ::= `{` entry-block? block* `}`
  // [x] - entry-block   ::= operation+
  //                    |
  //                    |  rewritten as
  //                   \/
  // [x] - region        ::= `{` operation* block* `}`

  def defineRegion(parseResult: (Seq[Operation], Seq[Block])): Region = {
    return new Region(blocks = parseResult._2)
  }

  // EntryBlock might break - take out if it does...
  def Region[$: P] = P(
    "{" ~ E(
      currentScope.switchWithChild
    ) ~ OperationPat.rep ~ Block.rep ~ "}" ~ E(
      currentScope.switchWithParent
    )
  ).map(defineRegion)

  // def EntryBlock[$: P] = P(OperationPat.rep(1))

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

  // [x ] function-type ::= (type | type-list-parens) `->` (type | type-list-parens)

  // // Type aliases
  // [x] type-alias-def ::= `!` alias-name `=` type
  // [x] type-alias ::= `!` alias-name

  def I32[$: P] = P("i32".!)
  def I64[$: P] = P("i64".!)
  def BuiltIn[$: P] = P(I32 | I64) // temporary BuiltIn

  def Type[$: P] = P(TypeAlias | BuiltIn).map((typeName: String) =>
    new Attribute(name = typeName)
  ) // shortened definition TODO: finish...

  def TypeListNoParens[$: P] =
    P(Type ~ ("," ~ Type).rep).map((types: (Attribute, Seq[Attribute])) =>
      types._1 +: types._2
    )

  def TypeListParens[$: P] = P(ClosedParens | "(" ~ TypeListNoParens ~ ")")

  def ClosedParens[$: P] = P("(" ~ ")").map(_ => Seq[Attribute]())

  def FunctionType[$: P] = P(
    (Type.map(Seq(_)) | TypeListParens) ~ "->" ~ (Type.map(
      Seq(_)
    ) | TypeListParens)
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

  def parseThis[A, B](
      text: String,
      pattern: fastparse.P[_] => fastparse.P[B] = { (x: fastparse.P[_]) =>
        TopLevel(x)
      }
  ): fastparse.Parsed[B] = {
    return parse(text, pattern)
  }
}

/////////////////////
// VERSION CONTROL //
/////////////////////

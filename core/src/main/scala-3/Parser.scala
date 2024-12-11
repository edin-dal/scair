package scair

import fastparse.*
import fastparse.Parsed.Failure
import fastparse.internal.Util
import scair.core.utils.Args
import scair.dialects.builtin.ModuleOp
import scair.ir.*

import java.lang.Float.parseFloat
import java.lang.Long.parseLong
import java.lang.Math.pow
import scala.annotation.switch
import scala.annotation.tailrec
import scala.collection.mutable

object Parser {

  /** Whitespace syntax that supports // line-comments, *without* /* */
    * comments, as is the case in the MLIR Language Spec.
    *
    * It's litteraly fastparse's JavaWhitespace with the /* */ states just
    * erased :)
    */
  implicit val whitespace: Whitespace = { implicit ctx: P[_] =>
    val input = ctx.input
    val startIndex = ctx.index
    @tailrec def rec(current: Int, state: Int): ParsingRun[Unit] = {
      if (!input.isReachable(current)) {
        if (state == 0 || state == 1) ctx.freshSuccessUnit(current)
        else ctx.freshSuccessUnit(current - 1)
      } else {
        val currentChar = input(current)
        (state: @switch) match {
          case 0 =>
            (currentChar: @switch) match {
              case ' ' | '\t' | '\n' | '\r' => rec(current + 1, state)
              case '/'                      => rec(current + 1, state = 2)
              case _                        => ctx.freshSuccessUnit(current)
            }
          case 1 =>
            rec(current + 1, state = if (currentChar == '\n') 0 else state)
          case 2 =>
            (currentChar: @switch) match {
              case '/' => rec(current + 1, state = 1)
              case _   => ctx.freshSuccessUnit(current - 1)
            }
        }
      }
    }
    rec(current = ctx.index, state = 0)
  }

  //////////////////////
  // COMMON FUNCTIONS //
  //////////////////////

  // Custom function wrapper that allows to Escape out of a pattern
  // to carry out custom computation
  def E[$: P](action: => Unit) = {
    action
    Pass(())
  }

  def giveBack[A](a: A) = {
    println(a)
    a
  }

  def optionlessSeq[A](option: Option[Seq[A]]): Seq[A] = option match {
    case None    => Seq[A]()
    case Some(x) => x
  }

  ///////////
  // SCOPE //
  ///////////

  object Scope {

    def defineValues(
        valueIdAndTypeList: Seq[(String, Attribute)]
    )(implicit
        scope: Scope
    ): ListType[Value[Attribute]] = {
      (for {
        (name, typ) <- valueIdAndTypeList
      } yield scope.valueMap.contains(name) match {
        case true =>
          throw new Exception(s"SSA Value cannot be defined twice %${name}")
        case false =>
          val value = new Value(typ = typ)
          scope.valueMap(name) = value
          value
      }).to(ListType)
    }

    def useValues(
        valueIdAndTypeList: Seq[(String, Attribute)]
    )(implicit
        scope: Scope
    ): (ListType[Value[Attribute]], ListType[(String, Attribute)]) = {
      var forwardRefSeq: ListType[(String, Attribute)] =
        ListType()
      var useValSeq: ListType[Value[Attribute]] =
        ListType()
      for {
        (name, typ) <- valueIdAndTypeList
      } yield !scope.valueMap.contains(name) match {
        case true =>
          val tuple = (name, typ)
          forwardRefSeq += tuple
        case false =>
          scope.valueMap(name).typ != typ match {
            case true =>
              throw new Exception(
                s"%$name use with type ${typ} but defined with type ${scope.valueMap(name).typ}"
              )
            case false =>
              useValSeq += scope.valueMap(name)
          }
      }
      return (useValSeq, forwardRefSeq)
    }

    def useValue(
        name: String,
        typ: Attribute
    )(implicit
        scope: Scope
    ): (ListType[Value[Attribute]], ListType[(String, Attribute)]) = {
      var forwardRefSeq: ListType[(String, Attribute)] =
        ListType()
      var useValSeq: ListType[Value[Attribute]] =
        ListType()
      !scope.valueMap.contains(name) match {
        case true =>
          val tuple = (name, typ)
          forwardRefSeq += tuple
        case false =>
          scope.valueMap(name).typ != typ match {
            case true =>
              throw new Exception(
                s"%$name use with type ${typ} but defined with type ${scope.valueMap(name).typ}"
              )
            case false =>
              useValSeq += scope.valueMap(name)
          }
      }
      return (useValSeq, forwardRefSeq)
    }

    def checkValueWaitlist()(implicit scope: Scope): Unit = {

      for ((operation, operands) <- scope.valueWaitlist) {

        val foundOperands: ListType[(String, Attribute)] = ListType()
        val operandList: ListType[Value[Attribute]] = ListType()

        for {
          (name, typ) <- operands
        } yield scope.valueMap
          .contains(
            name
          ) match {
          case true =>
            val tuple = (name, typ)
            val value = scope.valueMap(name)
            if (value.typ.name == typ.name) { // to be changed
              foundOperands += tuple
              operandList += value
            }
          case false =>
        }

        scope.valueWaitlist(operation) --= foundOperands

        if (scope.valueWaitlist(operation).length == 0) {
          scope.valueWaitlist -= operation
        }

        // adding Uses to each found operand
        val operandsLength = operation.operands.length

        // TO-DO: create a new OpOperands class specifically to close the API
        //        for operations operands
        for ((operand, i) <- operandList zip (0 to operandList.length)) {
          operand.uses += Use(operation, operandsLength + i)
        }

        operation.operands.appendAll(operandList)
      }

      if (scope.valueWaitlist.size > 0) {
        scope.parentScope match {
          case Some(x) => x.valueWaitlist ++= scope.valueWaitlist
          case None =>
            val opName = scope.valueWaitlist.head._1.name
            val operandName = scope.valueWaitlist.head._2(0)._1
            val operandTyp = scope.valueWaitlist.head._2(0)._2
            throw new Exception(
              s"Operand '${operandName}: ${operandTyp}' not defined within Scope in Operation '${opName}' "
            )
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

    def useBlocks(
        successorList: Seq[String]
    )(implicit
        scope: Scope
    ): (ListType[Block], ListType[String]) = {
      var forwardRefSeq: ListType[String] =
        ListType()
      var successorBlockSeq: ListType[Block] =
        ListType()
      for {
        name <- successorList
      } yield !scope.blockMap.contains(name) match {
        case true =>
          forwardRefSeq += name
        case false =>
          successorBlockSeq += scope.blockMap(name)
      }
      return (successorBlockSeq, forwardRefSeq)
    }

    // check block waitlist once you exit the local scope of a region,
    // as well as the global scope of the program at the end
    def checkBlockWaitlist()(implicit scope: Scope): Unit = {

      for ((operation, successors) <- scope.blockWaitlist) {

        val foundOperands: ListType[String] = ListType()
        val successorList: ListType[Block] = ListType()

        for {
          name <- successors
        } yield scope.blockMap
          .contains(
            name
          ) match {
          case true =>
            foundOperands += name
            successorList += scope.blockMap(name)
          case false =>
        }

        scope.blockWaitlist(operation) --= foundOperands

        if (scope.blockWaitlist(operation).length == 0) {
          scope.blockWaitlist -= operation
        }
        operation.successors.appendAll(successorList)
      }

      if (scope.blockWaitlist.size > 0) {
        scope.parentScope match {
          case Some(x) => x.blockWaitlist ++= scope.blockWaitlist
          case None =>
            throw new Exception(
              s"Successor ^${scope.blockWaitlist.head._2.head} not defined within Scope"
            )
        }
      }
    }
  }

  class Scope(
      var parentScope: Option[Scope] = None,
      var valueMap: mutable.Map[String, Value[Attribute]] =
        mutable.Map.empty[String, Value[Attribute]],
      var valueWaitlist: mutable.Map[Operation, ListType[
        (String, Attribute)
      ]] = mutable.Map.empty[Operation, ListType[(String, Attribute)]],
      var blockMap: mutable.Map[String, Block] =
        mutable.Map.empty[String, Block],
      var blockWaitlist: mutable.Map[Operation, ListType[String]] =
        mutable.Map.empty[Operation, ListType[String]]
  ) {

    // child starts off from the parents context
    def createChild(): Scope = {
      return new Scope(
        valueMap = valueMap.clone,
        blockMap = blockMap.clone,
        parentScope = Some(this)
      )
    }

    def switchWithParent(scope: Scope): Scope = parentScope match {
      case Some(x) =>
        Scope.checkValueWaitlist()(scope)
        Scope.checkBlockWaitlist()(scope)
        x
      case None =>
        scope
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

  val excludedCharacters: Set[Char] = Set('\"', '\n', '\f', '\u000B',
    '\r') // \u000B represents \v escape character

  def Digit[$: P] = P(CharIn("0-9").!)

  def HexDigit[$: P] = P(CharIn("0-9a-fA-F").!)

  def Letter[$: P] = P(CharIn("a-zA-Z").!)

  def IdPunct[$: P] = P(CharIn("$._\\-").!)

  def IntegerLiteral[$: P] = P(HexadecimalLiteral | DecimalLiteral)

  def DecimalLiteral[$: P] =
    P(Digit.rep(1).!).map((literal: String) => parseLong(literal))

  def HexadecimalLiteral[$: P] =
    P("0x" ~~ HexDigit.rep(1).!).map((hex: String) => parseLong(hex, 16))

  private def parseFloatNum(float: (String, String)): Double = {
    val number = parseFloat(float._1)
    val power = parseLong(float._2)
    return number * pow(10, power)
  }

  def FloatLiteral[$: P] = P(
    CharIn("\\-\\+").? ~~ (Digit.rep(1) ~~ "." ~~ Digit.rep(1)).!
      ~~ (CharIn("eE")
        ~~ (CharIn("\\-\\+").? ~~ Digit.rep(1)).!).?.map(_.getOrElse("0"))
  ).map(parseFloatNum(_)) // substituted [0-9]* with [0-9]+

  def notExcluded[$: P] = P(
    CharPred(char => !excludedCharacters.contains(char))
  )

  def StringLiteral[$: P] = P("\"" ~~ notExcluded.rep.! ~~ "\"")

  /////////////////
  // IDENTIFIERS //
  /////////////////

  // [x] bare-id ::= (letter|[_]) (letter|digit|[_$.])*
  // [ ] bare-id-list ::= bare-id (`,` bare-id)*
  // [x] value-id ::= `%` suffix-id
  // [x] alias-name :: = bare-id
  // [x] suffix-id ::= (digit+ | ((letter|id-punct) (letter|id-punct|digit)*))

  // [ ] symbol-ref-id ::= `@` (suffix-id | string-literal) // - redundant - (`::` symbol-ref-id)?
  // [ ] value-id-list ::= value-id (`,` value-id)*

  // // Uses of value, e.g. in an operand list to an operation.
  // [x] value-use ::= value-id (`#` decimal-literal)?
  // [x] value-use-list ::= value-use (`,` value-use)*

  def simplifyValueName(valueUse: (String, Option[Long])): String =
    valueUse match {
      case (name, Some(number)) => s"$name#$number"
      case (name, None)         => name
    }

  def BareId[$: P] = P(
    (Letter | "_") ~~ (Letter | Digit | CharIn("_$.")).rep
  ).!

  def ValueId[$: P] = P("%" ~~ SuffixId)

  def AliasName[$: P] = P(BareId)

  def SuffixId[$: P] = P(
    DecimalLiteral | (Letter | IdPunct) ~~ (Letter | IdPunct | Digit).rep
  ).!

  def SymbolRefId[$: P] = P("@" ~ (SuffixId | StringLiteral))

  def ValueUse[$: P] =
    P(ValueId ~ ("#" ~ DecimalLiteral).?).map(simplifyValueName)

  def ValueUseList[$: P] =
    P(ValueUse.rep(sep = ","))

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

  // [x] function-type ::= (type | type-list-parens) `->` (type | type-list-parens)

  def AttributeAlias[$: P] = P("#" ~ AliasName)

  ////////////////
  // OPERATIONS //
  ////////////////

  // [x] op-result-list        ::= op-result (`,` op-result)* `=`
  // [x] op-result             ::= value-id (`:` integer-literal)?
  // [x] successor-list        ::= `[` successor (`,` successor)* `]`
  // [x] successor             ::= caret-id (`:` block-arg-list)?
  // [x] trailing-location     ::= `loc` `(` location `)`

  def OpResultList[$: P] = P(
    OpResult.rep(1, sep = ",") ~ "="
  ).map((results: Seq[Seq[String]]) => results.flatten)

  def sequenceValues(value: (String, Option[Long])): Seq[String] =
    value match {
      case (name, Some(totalNo)) => (0 to totalNo.toInt).map(no => s"$name#$no")
      case (name, None)          => Seq(name)
    }

  def OpResult[$: P] =
    P(ValueId ~ (":" ~ DecimalLiteral).?).map(sequenceValues)

  def SuccessorList[$: P] = P("[" ~ Successor.rep(sep = ",") ~ "]")

  def Successor[$: P] = P(CaretId) // possibly shortened version

  def TrailingLocation[$: P] = P(
    "loc" ~ "(" ~ "unknown" ~ ")"
  ) // definition taken from xdsl attribute_parser.py v-0.19.0 line 1106

  ////////////
  // BLOCKS //
  ////////////

  // [x] - block-id        ::= caret-id
  // [x] - caret-id        ::= `^` suffix-id

  def BlockId[$: P] = P(CaretId)

  def CaretId[$: P] = P("^" ~ SuffixId)

  ///////////////////
  // DIALECT TYPES //
  ///////////////////

  // [x] - dialect-namespace      ::= bare-id

  // [x] - dialect-type           ::= `!` (opaque-dialect-type | pretty-dialect-type)
  // [x] - opaque-dialect-type    ::= dialect-namespace dialect-type-body
  // [x] - pretty-dialect-type    ::= dialect-namespace `.` pretty-dialect-type-lead-ident dialect-type-body?
  // [x] - pretty-dialect-type-lead-ident ::= `[A-Za-z][A-Za-z0-9._]*`

  // [x] - dialect-type-body      ::= `<` dialect-type-contents+ `>`
  // [x] - dialect-type-contents  ::= dialect-type-body
  //                             | `(` dialect-type-contents+ `)`
  //                             | `[` dialect-type-contents+ `]`
  //                             | `{` dialect-type-contents+ `}`
  //                             | [^\[<({\]>)}\0]+

  val excludedCharactersDTC: Set[Char] =
    Set('\\', '[', '<', '(', '{', '}', ')', '>', ']', '\u0000')

  def notExcludedDTC[$: P] = P(
    CharPred(char => !excludedCharacters.contains(char))
  )

  def DialectBareId[$: P] = P(
    (Letter | "_") ~~ (Letter | Digit | CharIn("_$")).rep
  ).!

  def DialectNamespace[$: P] = P(DialectBareId)

  def PrettyDialectReferenceName[$: P] = P(
    (DialectNamespace ~ "." ~ PrettyDialectTypeOrAttReferenceName)
  )

  def OpaqueDialectReferenceName[$: P] = P(
    (DialectNamespace ~ "<" ~ PrettyDialectTypeOrAttReferenceName)
  )

  def DialectReferenceName[$: P] = P(
    PrettyDialectReferenceName | OpaqueDialectReferenceName
  )

  def PrettyDialectTypeOrAttReferenceName[$: P] = P(
    (CharIn("a-zA-Z") ~~ CharsWhileIn("a-zA-Z0-9_")).!
  )

  // def OpaqueDialectType[$: P] = P(DialectNamespace ~ DialectTypeBody)

  // def DialectTypeBody[$: P] = P("<" ~ DialectTypeContents.rep(1) ~ ">")

  // def DialectTypeContents[$: P]: P[Any] = P(
  //   CharPred(char => !excludedCharactersDTC.contains(char)).rep.! |
  //     ("<" ~ DialectTypeContents.rep(1) ~ ">") |
  //     ("(" ~ DialectTypeContents.rep(1) ~ ")") |
  //     ("[" ~ DialectTypeContents.rep(1) ~ "]") |
  //     ("{" ~ DialectTypeContents.rep(1) ~ "}")
  // )

  //////////////////////////////
  // DIALECT ATTRIBUTE VALUES //
  //////////////////////////////

  // [x] - dialect-namespace ::= bare-id

  // [x] - dialect-attribute ::= `#` (opaque-dialect-attribute | pretty-dialect-attribute)
  // [x] - opaque-dialect-attribute ::= dialect-namespace dialect-attribute-body
  // [x] - pretty-dialect-attribute ::= dialect-namespace `.` pretty-dialect-attribute-lead-ident dialect-attribute-body?
  // [x] - pretty-dialect-attribute-lead-ident ::= `[A-Za-z][A-Za-z0-9._]*`

  // [x] - dialect-attribute-body ::= `<` dialect-attribute-contents+ `>`
  // [x] - dialect-attribute-contents ::= dialect-attribute-body
  //                             | `(` dialect-attribute-contents+ `)`
  //                             | `[` dialect-attribute-contents+ `]`
  //                             | `{` dialect-attribute-contents+ `}`
  //                             | [^\[<({\]>)}\0]+

  // def DialectAttribute[$: P] = P(
  //   "!" ~ (PrettyDialectAttribute | OpaqueDialectAttribute)
  // )

  // def PrettyDialectAttribute[$: P] = P(
  //   DialectNamespace ~ "." ~ PrettyDialectAttributeLeadIdent
  // )

  // def PrettyDialectAttributeLeadIdent[$: P] = P(
  //   (CharIn("a-zA-Z").! ~ CharIn("a-zA-Z0-9._").rep).!
  // )

  // def OpaqueDialectAttribute[$: P] = P(DialectNamespace ~ DialectAttributeBody)

  // def DialectAttributeBody[$: P] = P(
  //   "<" ~ DialectAttributeContents.rep(1) ~ ">"
  // )

  // def DialectAttributeContents[$: P]: P[Any] = P(
  //   CharPred(char => !excludedCharactersDTC.contains(char)).rep.! |
  //     ("<" ~ DialectAttributeContents.rep(1) ~ ">") |
  //     ("(" ~ DialectAttributeContents.rep(1) ~ ")") |
  //     ("[" ~ DialectAttributeContents.rep(1) ~ "]") |
  //     ("{" ~ DialectAttributeContents.rep(1) ~ "}")
  // )

}

class Parser(val context: MLContext, val args: Args = Args())
    extends AttrParser(context) {

  import Parser._

  implicit val whitespace: Whitespace = Parser.whitespace

  def error(failure: Failure) =
    val traced = failure.extra.traced
    val msg =
      s"Parse error at ${args.input.getOrElse("-")}:${failure.extra.input
          .prettyIndex(failure.index)}:\n\n${failure.extra.trace().aggregateMsg}"

    if args.parsing_diagnostics then msg
    else {
      Console.err.println(msg)
      sys.exit(1)
    }

  implicit var currentScope: Scope = new Scope()

  def enterLocalRegion = {
    currentScope = currentScope.createChild()
  }

  def enterParentRegion = {
    currentScope = currentScope.switchWithParent(currentScope)
  }

  def defineBlockValues(valueIdAndTypeList: Seq[(String, Attribute)]) = {
    Scope.defineValues(valueIdAndTypeList)
  }

  //////////////////////////
  // TOP LEVEL PRODUCTION //
  //////////////////////////

  // [x] toplevel := (operation | attribute-alias-def | type-alias-def)*
  // shortened definition TODO: finish...

  def TopLevel[$: P]: P[Operation] = P(
    Start ~ (ModuleOp.parse(this) | Operations(0)) ~ E({
      Scope.checkValueWaitlist()
      Scope.checkBlockWaitlist()
    }) ~ End
  ).map((toplevel: Operation | ListType[Operation]) =>
    toplevel match {
      case x: ModuleOp => x
      case y: ListType[Operation] =>
        val block = new Block(operations = y)
        val region = new Region(blocks = Seq(block))
        val moduleOp = new ModuleOp(regions = ListType(region))

        for (op <- y) op.container_block = Some(block)
        block.container_region = Some(region)
        region.container_operation = Some(moduleOp)

        moduleOp
    }
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
  // [x] region-list           ::= `(` region (`,` region)* `)`

  def verifyCustomOp(
      opGen: (
          ListType[Value[Attribute]] /* = operands */,
          ListType[Block] /* = successors */,
          ListType[Value[Attribute]] /* = results */,
          ListType[Region] /* = regions */,
          DictType[String, Attribute] /* = dictProps */,
          DictType[String, Attribute] /* = dictAttrs */
      ) => Operation,
      opName: String,
      operandNames: Seq[String] = Seq(),
      operandTypes: Seq[Attribute] = Seq(),
      successors: Seq[String] = Seq(),
      resultNames: Seq[String] = Seq(),
      resultTypes: Seq[Attribute] = Seq(),
      regions: Seq[Region] = Seq(),
      dictProps: Seq[(String, Attribute)] = Seq(),
      dictAttrs: Seq[(String, Attribute)] = Seq(),
      noForwardOperandRef: Int = 0
  ): Operation = {

    val regionss = regions.to(ListType)

    if (resultNames.length != resultTypes.length) {
      throw new Exception(
        s"Number of results does not match the number of the corresponding result types in \"${opName}\"."
      )
    }

    val dictPropertiesMap: DictType[String, Attribute] =
      DictType[String, Attribute](dictProps.map({ case (x, y) => x -> y }): _*)

    if (dictProps.length != dictPropertiesMap.size) {
      throw new Exception(
        "Dictionary Properties names in Operation " + opName + " are cloned."
      )
    }

    val dictAttributesMap: DictType[String, Attribute] =
      DictType[String, Attribute](dictAttrs.map({ case (x, y) => x -> y }): _*)

    if (dictAttrs.length != dictAttributesMap.size) {
      throw new Exception(
        "Dictionary Properties names in Operation " + opName + " are cloned."
      )
    }

    val resultss: ListType[Value[Attribute]] =
      Scope.defineValues(resultNames zip resultTypes)

    val useAndRefBlockSeqs: (ListType[Block], ListType[String]) =
      Scope.useBlocks(successors)

    // plaster solution for custom parsing
    if (noForwardOperandRef == 1) {

      val operandValues: ListType[Value[Attribute]] = (
        try {
          for { name <- operandNames } yield currentScope.valueMap(name)
        } catch {
          case e: Exception =>
            throw new Exception(
              s"Operands for Operation: \"${opName}\" must be pre-defined."
            )
        }
      ).to(ListType)

      val op: Operation = opGen(
        operandValues,
        useAndRefBlockSeqs._1,
        resultss,
        regionss,
        dictPropertiesMap,
        dictAttributesMap
      )

      for ((operand, i) <- operandValues zip (0 to operandValues.length)) {
        operand.uses += Use(op, i)
      }

      if (useAndRefBlockSeqs._2.length > 0) {
        currentScope.blockWaitlist += op -> useAndRefBlockSeqs._2
      }

      return op

    } else {

      if (operandNames.length != operandTypes.length) {
        throw new Exception(
          s"Number of operands does not match the number of the corresponding operand types in \"${opName}\"."
        )
      }

      val useAndRefValueSeqs
          : (ListType[Value[Attribute]], ListType[(String, Attribute)]) =
        Scope.useValues(operandNames zip operandTypes)

      val op: Operation = opGen(
        useAndRefValueSeqs._1,
        useAndRefBlockSeqs._1,
        resultss,
        regionss,
        dictPropertiesMap,
        dictAttributesMap
      )

      for (
        (operand, i) <-
          useAndRefValueSeqs._1 zip (0 to useAndRefValueSeqs._1.length)
      ) {
        operand.uses += Use(op, i)
      }

      if (useAndRefValueSeqs._2.length > 0) {
        currentScope.valueWaitlist += op -> useAndRefValueSeqs._2
      }
      if (useAndRefBlockSeqs._2.length > 0) {
        currentScope.blockWaitlist += op -> useAndRefBlockSeqs._2
      }

      return op
    }
  }

  //  results      name     operands   successors  dictprops  regions  dictattr  (op types, res types)
  def generateOperation(
      operation: (
          Seq[String],
          (
              String,
              Seq[String],
              Seq[String],
              Seq[(String, Attribute)],
              Seq[Region],
              Seq[(String, Attribute)],
              (Seq[Attribute], Seq[Attribute])
          )
      )
  ): Operation = {

    val results: Seq[String] = operation._1
    val opName = operation._2._1
    val operands: Seq[String] = operation._2._2
    val successors: Seq[String] = operation._2._3
    val dictProperties: Seq[(String, Attribute)] = operation._2._4
    val regions: ListType[Region] = operation._2._5.to(ListType)
    val dictAttributes: Seq[(String, Attribute)] = operation._2._6
    val resultsTypes = operation._2._7._2
    val operandsTypes = operation._2._7._1

    if (results.length != resultsTypes.length) {
      throw new Exception(
        s"Number of results does not match the number of the corresponding result types in \"${opName}\"."
      )
    }

    if (operands.length != operandsTypes.length) {
      throw new Exception(
        s"Number of operands does not match the number of the corresponding operand types in \"${opName}\"."
      )
    }

    val dictPropertiesMap: DictType[String, Attribute] =
      DictType[String, Attribute](
        dictProperties.map({ case (x, y) => x -> y }): _*
      )

    if (dictProperties.length != dictPropertiesMap.size) {
      throw new Exception(
        "Dictionary Properties names in Operation " + opName + " are cloned."
      )
    }

    val dictAttributesMap: DictType[String, Attribute] =
      DictType[String, Attribute](
        dictAttributes.map({ case (x, y) => x -> y }): _*
      )

    if (dictAttributes.length != dictAttributesMap.size) {
      throw new Exception(
        "Dictionary Properties names in Operation " + opName + " are cloned."
      )
    }

    val resultss: ListType[Value[Attribute]] =
      Scope.defineValues(results zip resultsTypes)

    val useAndRefValueSeqs
        : (ListType[Value[Attribute]], ListType[(String, Attribute)]) =
      Scope.useValues(operands zip operandsTypes)

    val useAndRefBlockSeqs: (ListType[Block], ListType[String]) =
      Scope.useBlocks(successors)

    val opObject: Option[OperationObject] = ctx.getOperation(opName)

    val op = opObject match {
      case Some(x) =>
        x.constructOp(
          operands = useAndRefValueSeqs._1,
          successors = useAndRefBlockSeqs._1,
          dictionaryProperties = dictPropertiesMap,
          results = resultss,
          dictionaryAttributes = dictAttributesMap,
          regions = regions
        )
      case None =>
        if args.allow_unregistered then
          new UnregisteredOperation(
            name = opName,
            operands = useAndRefValueSeqs._1,
            successors = useAndRefBlockSeqs._1,
            dictionaryProperties = dictPropertiesMap,
            results = resultss,
            dictionaryAttributes = dictAttributesMap,
            regions = regions
          )
        else
          throw new Exception(
            s"Operation ${opName} is not registered. If this is intended, use `--allow-unregistered-dialect`"
          )
    }

    // adding uses for known operands
    for (
      (operand, i) <-
        useAndRefValueSeqs._1 zip (0 to useAndRefValueSeqs._1.length)
    ) {
      operand.uses += Use(op, i)
    }
    if (useAndRefValueSeqs._2.length > 0) {
      currentScope.valueWaitlist += op -> useAndRefValueSeqs._2
    }
    if (useAndRefBlockSeqs._2.length > 0) {
      currentScope.blockWaitlist += op -> useAndRefBlockSeqs._2
    }

    return op
  }

  def Operations[$: P](at_least_this_many: Int = 0): P[ListType[Operation]] =
    P(OperationPat.rep(at_least_this_many).map(_.to(ListType)))

  def OperationPat[$: P]: P[Operation] = P(
    OpResultList.?./.map(
      optionlessSeq
    ).flatMap(Op(_)) ~/ TrailingLocation.?
  )

  def Op[$: P](resName: Seq[String]) = P(
    GenericOperation(resName) | CustomOperation(resName)
  ).map(op => {
    for (region <- op.regions) region.container_operation = Some(op)
    op
  })

  def GenericOperation[$: P](resNames: Seq[String]) = P(
    StringLiteral ~/ "(" ~ ValueUseList.?.map(optionlessSeq) ~ ")"
      ~/ SuccessorList.?.map(optionlessSeq)
      ~/ DictionaryProperties.?.map(optionlessSeq)
      ~/ RegionList.?.map(optionlessSeq)
      ~/ DictionaryAttribute.?.map(optionlessSeq) ~/ ":" ~/ FunctionType
  ).map(generateOperation(resNames, _))

  def CustomOperation[$: P](resNames: Seq[String]) = P(
    PrettyDialectReferenceName.flatMap { (x: String, y: String) =>
      ctx.getOperation(s"${x}.${y}") match {
        case Some(y) =>
          y.parse(resNames, this)
        case None =>
          throw new Exception(
            s"Operation ${x} is not defined in any supported Dialect."
          )
      }
    }
  )

  def RegionList[$: P] =
    P("(" ~ Region.rep(sep = ",") ~ ")")

  ////////////
  // BLOCKS //
  ////////////

  // [x] - block           ::= block-label operation+
  // [x] - block-label     ::= block-id block-arg-list? `:`

  def createBlock(
      //            name    argument     operations
      uncutBlock: (String, ListType[Value[Attribute]], ListType[Operation])
  ): Block = {
    val newBlock = new Block(
      operations = uncutBlock._3,
      arguments = uncutBlock._2
    )
    for (op <- newBlock.operations) op.container_block = Some(newBlock)
    Scope.defineBlock(uncutBlock._1, newBlock)
    return newBlock
  }

  def Block[$: P] =
    P(BlockLabel ~ Operations(0)).map(createBlock)

  def BlockLabel[$: P] = P(
    BlockId ~ BlockArgList.?.map(optionlessSeq)
      .map(defineBlockValues) ~ ":"
  )

  /////////////
  // REGIONS //
  /////////////

  // [x] - region        ::= `{` entry-block? block* `}`
  // [x] - entry-block   ::= operation+
  //                    |
  //                    |  rewritten as
  //                   \/
  // [x] - region        ::= `{` operation* block* `}`

  def defineRegion(parseResult: (ListType[Operation], Seq[Block])): Region = {
    return parseResult._1.length match {
      case 0 =>
        val region = new Region(blocks = parseResult._2)
        for (block <- region.blocks) block.container_region = Some(region)
        region
      case _ =>
        val startblock =
          new Block(operations = parseResult._1, arguments = ListType())
        val region = new Region(blocks = startblock +: parseResult._2)
        for (block <- region.blocks) block.container_region = Some(region)
        region
    }
  }

  // EntryBlock might break - take out if it does...
  def Region[$: P] = P(
    "{" ~ E(
      { enterLocalRegion }
    ) ~ Operations(0) ~ Block.rep ~ "}" ~ E(
      { enterParentRegion }
    )
  ).map(defineRegion)

  // def EntryBlock[$: P] = P(OperationPat.rep(1))

  // // Type aliases
  // [x] type-alias-def ::= `!` alias-name `=` type
  // [x] type-alias ::= `!` alias-name

  def TypeAliasDef[$: P] = P(
    "!" ~~ AliasName ~ "=" ~ Type
  )

  def TypeAlias[$: P] = P("!" ~~ AliasName)

  ////////////////
  // ATTRIBUTES //
  ////////////////

  // // Attribute Value Aliases
  // [x] - attribute-alias-def ::= `#` alias-name `=` attribute-value
  // [x] - attribute-alias ::= `#` alias-name

  def AttributeAliasDef[$: P] = P(
    "#" ~ AliasName ~ "=" ~ AttributeValue
  )

  // [x] - value-id-and-type ::= value-id `:` type

  // // Non-empty list of names and types.
  // [x] - value-id-and-type-list ::= value-id-and-type (`,` value-id-and-type)*

  // [x] - block-arg-list ::= `(` value-id-and-type-list? `)`

  def ValueIdAndType[$: P] = P(ValueId ~ ":" ~ Type)

  def ValueIdAndTypeList[$: P] =
    P(ValueIdAndType.rep(sep = ","))

  def BlockArgList[$: P] =
    P("(" ~ ValueIdAndTypeList.? ~ ")").map(optionlessSeq)

  // [x] dictionary-properties ::= `<` dictionary-attribute `>`
  // [x] dictionary-attribute  ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`

  def DictionaryProperties[$: P] = P(
    "<" ~ DictionaryAttribute ~ ">"
  )

  def DictionaryAttribute[$: P] = P(
    "{" ~ AttributeEntry.rep(sep = ",") ~ "}"
  )

  ////////////////////
  // PARSE FUNCTION //
  ////////////////////

  def parseThis[A, B](
      text: String,
      pattern: fastparse.P[_] => fastparse.P[B] = { (x: fastparse.P[_]) =>
        TopLevel(x)
      }
  ): fastparse.Parsed[B] = {
    return parse(text, pattern)
  }
}

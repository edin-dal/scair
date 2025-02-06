package scair

import fastparse.*
import fastparse.Implicits.Repeater
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

// ██████╗░ ░█████╗░ ██████╗░ ░██████╗ ███████╗ ██████╗░
// ██╔══██╗ ██╔══██╗ ██╔══██╗ ██╔════╝ ██╔════╝ ██╔══██╗
// ██████╔╝ ███████║ ██████╔╝ ╚█████╗░ █████╗░░ ██████╔╝
// ██╔═══╝░ ██╔══██║ ██╔══██╗ ░╚═══██╗ ██╔══╝░░ ██╔══██╗
// ██║░░░░░ ██║░░██║ ██║░░██║ ██████╔╝ ███████╗ ██║░░██║
// ╚═╝░░░░░ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ╚═════╝░ ╚══════╝ ╚═╝░░╚═╝

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

  /*≡==--==≡≡≡≡==--=≡≡*\
  || COMMON FUNCTIONS ||
  \*≡==---==≡≡==---==≡*/

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

  extension [T](inline p: P[T])

    /** Make the parser optional, parsing defaults if otherwise failing.
      *
      * @todo:
      *   Figure out dark implicit magic to figure out magically that the
      *   default default is "T()".
      *
      * @param default
      *   The default value to use if the parser fails.
      * @return
      *   An optional parser, defaulting to default.
      */
    inline def orElse[$: P](inline default: T): P[T] = P(
      p.?.map(_.getOrElse(default))
    )

    /** Like fastparse's flatMapX but capturing exceptions as standard parse
      * errors.
      *
      * @note
      *   flatMapX because it often yields more nat ural error positions.
      *
      * @param f
      *   The function to apply to the parsed value.
      * @return
      *   A parser that applies f to the parsed value, catching exceptions and
      *   turning them into parse errors.
      */
    inline def flatMapTry[$: P, V](inline f: T => P[V]): P[V] = P(
      p.flatMapX(parsed =>
        try {
          f(parsed)
        } catch {
          case e: Exception =>
            Fail(e.getMessage())
        }
      )
    )

    /** Like fastparse's mapX but capturing exceptions as standard parse errors.
      *
      * @note
      *   flatMapX because it often yields more nat ural error positions.
      *
      * @param f
      *   The function to apply to the parsed value.
      * @return
      *   A parser that applies f to the parsed value, catching exceptions and
      *   turning them into parse errors.
      */
    inline def mapTry[$: P, V](inline f: T => V): P[V] = P(
      p.flatMapX(parsed =>
        try {
          Pass(f(parsed))
        } catch {
          case e: Exception =>
            Fail(e.getMessage())
        }
      )
    )

  /*≡==--==≡≡≡==--=≡≡*\
  ||      SCOPE      ||
  \*≡==---==≡==---==≡*/

  object Scope {}

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

    def defineValues(
        valueIdAndTypeList: Seq[(String, Value[Attribute])]
    ): Unit = {
      valueIdAndTypeList.map((name, value) =>
        valueMap.contains(name) match {
          case true =>
            throw new Exception(s"SSA Value cannot be defined twice %${name}")
          case false =>
            valueMap(name) = value
        }
      )
    }

    def useValues(
        valueIdAndTypeList: Seq[(String, Attribute)]
    ): (ListType[Value[Attribute]], ListType[(String, Attribute)]) = {
      var forwardRefSeq: ListType[(String, Attribute)] =
        ListType()
      var useValSeq: ListType[Value[Attribute]] =
        ListType()
      for {
        (name, typ) <- valueIdAndTypeList
      } yield !valueMap.contains(name) match {
        case true =>
          val tuple = (name, typ)
          forwardRefSeq += tuple
        case false =>
          valueMap(name).typ != typ match {
            case true =>
              throw new Exception(
                s"%$name use with type ${typ} but defined with type ${valueMap(name).typ}"
              )
            case false =>
              useValSeq += valueMap(name)
          }
      }
      return (useValSeq, forwardRefSeq)
    }

    def useValue(
        name: String,
        typ: Attribute
    ): (ListType[Value[Attribute]], ListType[(String, Attribute)]) = {
      var forwardRefSeq: ListType[(String, Attribute)] =
        ListType()
      var useValSeq: ListType[Value[Attribute]] =
        ListType()
      !valueMap.contains(name) match {
        case true =>
          val tuple = (name, typ)
          forwardRefSeq += tuple
        case false =>
          valueMap(name).typ != typ match {
            case true =>
              throw new Exception(
                s"%$name use with type ${typ} but defined with type ${valueMap(name).typ}"
              )
            case false =>
              useValSeq += valueMap(name)
          }
      }
      return (useValSeq, forwardRefSeq)
    }

    def checkValueWaitlist(): Unit = {

      for ((operation, operands) <- valueWaitlist) {

        val foundOperands: ListType[(String, Attribute)] = ListType()
        val operandList: ListType[Value[Attribute]] = ListType()

        for {
          (name, typ) <- operands
        } yield valueMap
          .contains(
            name
          ) match {
          case true =>
            val tuple = (name, typ)
            val value = valueMap(name)
            if (value.typ.name == typ.name) { // to be changed
              foundOperands += tuple
              operandList += value
            }
          case false =>
        }

        valueWaitlist(operation) --= foundOperands

        if (valueWaitlist(operation).length == 0) {
          valueWaitlist -= operation
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

      if (valueWaitlist.size > 0) {
        parentScope match {
          case Some(x) => x.valueWaitlist ++= valueWaitlist
          case None =>
            val opName = valueWaitlist.head._1.name
            val operandName = valueWaitlist.head._2(0)._1
            val operandTyp = valueWaitlist.head._2(0)._2
            throw new Exception(
              s"Operand '${operandName}: ${operandTyp}' not defined within Scope in Operation '${opName}'\n${valueMap}"
            )
        }
      }
    }

    def defineBlock(
        blockName: String,
        block: Block
    ): Unit = blockMap.contains(blockName) match {
      case true =>
        throw new Exception(
          s"Block cannot be defined twice within the same scope - ^${blockName}"
        )
      case false =>
        blockMap(blockName) = block
    }

    def useBlocks(
        successorList: Seq[String]
    ): (ListType[Block], ListType[String]) = {
      var forwardRefSeq: ListType[String] =
        ListType()
      var successorBlockSeq: ListType[Block] =
        ListType()
      for {
        name <- successorList
      } yield !blockMap.contains(name) match {
        case true =>
          forwardRefSeq += name
        case false =>
          successorBlockSeq += blockMap(name)
      }
      return (successorBlockSeq, forwardRefSeq)
    }

    // check block waitlist once you exit the local scope of a region,
    // as well as the global scope of the program at the end
    def checkBlockWaitlist(): Unit = {

      for ((operation, successors) <- blockWaitlist) {

        val foundOperands: ListType[String] = ListType()
        val successorList: ListType[Block] = ListType()

        for {
          name <- successors
        } yield blockMap
          .contains(
            name
          ) match {
          case true =>
            foundOperands += name
            successorList += blockMap(name)
          case false =>
        }

        blockWaitlist(operation) --= foundOperands

        if (blockWaitlist(operation).length == 0) {
          blockWaitlist -= operation
        }
        operation.successors.appendAll(successorList)
      }

      if (blockWaitlist.size > 0) {
        parentScope match {
          case Some(x) => x.blockWaitlist ++= blockWaitlist
          case None =>
            throw new Exception(
              s"Successor ^${blockWaitlist.head._2.head} not defined within Scope"
            )
        }
      }
    }

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
        scope.checkValueWaitlist()
        scope.checkBlockWaitlist()
        x
      case None =>
        scope
    }

  }

  /*≡==--==≡≡≡==--=≡≡*\
  ||  COMMON SYNTAX  ||
  \*≡==---==≡==---==≡*/

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
    P(("-" | "+").?.! ~ Digit.repX(1).!).map((sign: String, literal: String) =>
      parseLong(sign + literal)
    )

  def HexadecimalLiteral[$: P] =
    P("0x" ~~ HexDigit.repX(1).!).map((hex: String) => parseLong(hex, 16))

  private def parseFloatNum(float: (String, String)): Double = {
    val number = parseFloat(float._1)
    val power = parseLong(float._2)
    return number * pow(10, power)
  }

  def FloatLiteral[$: P] = P(
    CharIn("\\-\\+").? ~~ (Digit.repX(1) ~~ "." ~~ Digit.repX(1)).!
      ~~ (CharIn("eE")
        ~~ (CharIn("\\-\\+").? ~~ Digit.repX(1)).!).orElse("0")
  ).map(parseFloatNum(_)) // substituted [0-9]* with [0-9]+

  def notExcluded[$: P] = P(
    CharPred(char => !excludedCharacters.contains(char))
  )

  def StringLiteral[$: P] = P("\"" ~~ notExcluded.rep.! ~~ "\"")

  /*≡==--==≡≡≡==--=≡≡*\
  ||   IDENTIFIERS   ||
  \*≡==---==≡==---==≡*/

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
    (Letter | "_") ~~ (Letter | Digit | CharIn("_$.")).repX
  ).!

  def ValueId[$: P] = P("%" ~~ SuffixId)

  def AliasName[$: P] = P(BareId)

  def SuffixId[$: P] = P(
    DecimalLiteral | (Letter | IdPunct) ~~ (Letter | IdPunct | Digit).repX
  ).!

  def SymbolRefId[$: P] = P("@" ~~ (SuffixId | StringLiteral))

  def ValueUse[$: P] =
    P(ValueId ~ ("#" ~~ DecimalLiteral).?).map(simplifyValueName)

  def ValueUseList[$: P] =
    P(ValueUse.rep(sep = ","))

  /*≡==--==≡≡≡==--=≡≡*\
  ||      TYPES      ||
  \*≡==---==≡==---==≡*/

  // [x] type ::= type-alias | dialect-type | builtin-type

  // [x] type-list-no-parens ::=  type (`,` type)*
  // [x] type-list-parens ::= `(` `)` | `(` type-list-no-parens `)`

  // // This is a common way to refer to a value with a specified type.
  // [ ] ssa-use-and-type ::= ssa-use `:` type
  // [ ] ssa-use ::= value-use

  // // Non-empty list of names and types.
  // [ ] ssa-use-and-type-list ::= ssa-use-and-type (`,` ssa-use-and-type)*

  // [x] function-type ::= (type | type-list-parens) `->` (type | type-list-parens)

  def AttributeAlias[$: P] = P("#" ~~ AliasName)

  /*≡==--==≡≡≡≡==--=≡≡*\
  ||    OPERATIONS    ||
  \*≡==---==≡≡==---==≡*/

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
      case (name, Some(totalNo)) =>
        (0 to (totalNo.toInt - 1)).map(no => s"$name#$no")
      case (name, None) => Seq(name)
    }

  def OpResult[$: P] =
    P(ValueId ~ (":" ~ DecimalLiteral).?).map(sequenceValues)

  def SuccessorList[$: P] = P("[" ~ Successor.rep(sep = ",") ~ "]")

  def Successor[$: P] = P(CaretId) // possibly shortened version

  def TrailingLocation[$: P] = P(
    "loc" ~ "(" ~ "unknown" ~ ")"
  ) // definition taken from xdsl attribute_parser.py v-0.19.0 line 1106

  /*≡==--==≡≡≡≡==--=≡≡*\
  ||      BLOCKS      ||
  \*≡==---==≡≡==---==≡*/

  // [x] - block-id        ::= caret-id
  // [x] - caret-id        ::= `^` suffix-id

  def BlockId[$: P] = P(CaretId)

  def CaretId[$: P] = P("^" ~ SuffixId)

  /*≡==--==≡≡≡==--=≡≡*\
  ||  DIALECT TYPES  ||
  \*≡==---==≡==---==≡*/

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
    (Letter | "_") ~~ (Letter | Digit | CharIn("_$")).repX
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

}

/*≡==--==≡≡≡≡≡≡≡≡==--=≡≡*\
||     PARSER CLASS     ||
\*≡==---==≡≡≡≡≡≡==---==≡*/

class Parser(val context: MLContext, val args: Args = Args())
    extends AttrParser(context) {

  import Parser._

  implicit val whitespace: Whitespace = Parser.whitespace

  def error(failure: Failure) =
    // .trace() below reparses from the start with more bookkeeping to provide helpful
    // context for the error message.
    // We do this very non-functional bookkeeping in currentScope ourselves, which
    // is disfunctional with this behaviour; it then reparses everything with the state
    // it already had at the time of the catched error!
    // This is a workaround to get the error message with the correct state.
    // TODO: More functional and fastparse-compatible state handling!
    currentScope = new Scope()

    // Reparse for more context on error.
    val traced = failure.trace()
    // Get the line and column of the error.
    val prettyIndex =
      traced.input.prettyIndex(traced.index).split(":").map(_.toInt)
    val (line, col) = (prettyIndex(0), prettyIndex(1))

    // Get the error's line's content
    val length = traced.input.length
    val input_line = traced.input.slice(0, length).split("\n")(line - 1)

    // Build a visual indicator of where the error is.
    val indicator = " " * (col - 1) + "^"

    // Build the error message.
    val msg =
      s"Parse error at ${args.input.getOrElse("-")}:$line:$col:\n\n$input_line\n$indicator\n${traced.label}"

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

  /*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
  || TOP LEVEL PRODUCTION  ||
  \*≡==---==≡≡≡≡≡≡≡==---==≡*/

  // [x] toplevel := (operation | attribute-alias-def | type-alias-def)*
  // shortened definition TODO: finish...

  def TopLevel[$: P]: P[Operation] = P(
    Start ~ (Operations(0)) ~ E({
      currentScope.checkValueWaitlist()
      currentScope.checkBlockWaitlist()
    }) ~ End
  ).map((toplevel: ListType[Operation]) =>
    toplevel.toList match {
      case (head: ModuleOp) :: Nil => head
      case _ =>
        val block = new Block(operations = toplevel)
        val region = new Region(blocks = Seq(block))
        val moduleOp = new ModuleOp(regions = ListType(region))

        for (op <- toplevel) op.container_block = Some(block)
        block.container_region = Some(region)
        region.container_operation = Some(moduleOp)

        moduleOp

    }
  )

  /*≡==--==≡≡≡≡==--=≡≡*\
  ||    OPERATIONS    ||
  \*≡==---==≡≡==---==≡*/

  // [x] operation             ::= op-result-list? (generic-operation | custom-operation)
  //                         trailing-location?
  // [x] generic-operation     ::= string-literal `(` value-use-list? `)`  successor-list?
  //                         dictionary-properties? region-list? dictionary-attribute?
  //                         `:` function-type
  // [ ] custom-operation      ::= bare-id custom-operation-format
  // [x] region-list           ::= `(` region (`,` region)* `)`

  //  results      name     operands   successors  dictprops  regions  dictattr  (op types, res types)

  /** Generates an operation based on the provided parameters.
    *
    * @param opName
    *   The name of the operation to generate.
    * @param operandsNames
    *   A sequence of operand names. Defaults to an empty sequence.
    * @param successorsNames
    *   A sequence of successor names. Defaults to an empty sequence.
    * @param properties
    *   A dictionary of properties for the operation. Defaults to an empty
    *   dictionary.
    * @param regions
    *   A sequence of regions for the operation. Defaults to an empty sequence.
    * @param attributes
    *   A dictionary of attributes for the operation. Defaults to an empty
    *   dictionary.
    * @param resultsTypes
    *   A sequence of result types for the operation. Defaults to an empty
    *   sequence.
    * @param operandsTypes
    *   A sequence of operand types. Defaults to an empty sequence.
    * @return
    *   The generated operation.
    */
  def generateOperation(
      opName: String,
      operandsNames: Seq[String] = Seq(),
      successorsNames: Seq[String] = Seq(),
      properties: DictType[String, Attribute] = DictType(),
      regions: Seq[Region] = Seq(),
      attributes: DictType[String, Attribute] = DictType(),
      resultsTypes: Seq[Attribute] = Seq(),
      operandsTypes: Seq[Attribute] = Seq()
  ): Operation = {

    if (operandsNames.length != operandsTypes.length) {
      throw new Exception(
        s"Number of operands does not match the number of the corresponding operand types in \"${opName}\"."
      )
    }

    val useAndRefValueSeqs
        : (ListType[Value[Attribute]], ListType[(String, Attribute)]) =
      currentScope.useValues(operandsNames zip operandsTypes)

    val useAndRefBlockSeqs: (ListType[Block], ListType[String]) =
      currentScope.useBlocks(successorsNames)

    val opObject: Option[OperationObject] = ctx.getOperation(opName)

    val op = opObject match {
      case Some(x) =>
        x.constructOp(
          operands = useAndRefValueSeqs._1,
          successors = useAndRefBlockSeqs._1,
          dictionaryProperties = properties,
          results_types = ListType.from(resultsTypes),
          dictionaryAttributes = attributes,
          regions = ListType.from(regions)
        )

      case None =>
        if args.allow_unregistered then
          new UnregisteredOperation(
            name = opName,
            operands = useAndRefValueSeqs._1,
            successors = useAndRefBlockSeqs._1,
            dictionaryProperties = properties,
            results_types = ListType.from(resultsTypes),
            dictionaryAttributes = attributes,
            regions = ListType.from(regions)
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
    OpResultList.orElse(Seq())./.flatMap(Op(_)) ~/ TrailingLocation.?
  )

  def Op[$: P](resNames: Seq[String]) = P(
    GenericOperation(resNames) | CustomOperation(resNames)
  ).mapTry(op => {
    if (resNames.length != op.results.length) {
      throw new Exception(
        s"Number of results (${resNames.length}) does not match the number of the corresponding result types (${resNames.length}) in \"${op.name}\"."
      )
    }
    currentScope.defineValues(resNames zip op.results)
    for (region <- op.regions) region.container_operation = Some(op)
    op
  })

  def GenericOperation[$: P](resNames: Seq[String]) = P(
    (StringLiteral ~/ "(" ~ ValueUseList.orElse(Seq()) ~ ")"
      ~/ SuccessorList.orElse(Seq())
      ~/ DictionaryProperties.orElse(DictType.empty)
      ~/ RegionList.orElse(Seq())
      ~/ (DictionaryAttribute).orElse(DictType.empty) ~/ ":" ~/ FunctionType)
      .mapTry(
        (
            (
                opName: String,
                operandsNames: Seq[String],
                successorsNames: Seq[String],
                properties: DictType[String, Attribute],
                regions: Seq[Region],
                attributes: DictType[String, Attribute],
                operandsAndResultsTypes: (Seq[Attribute], Seq[Attribute])
            ) =>
              generateOperation(
                opName,
                operandsNames,
                successorsNames,
                properties,
                regions,
                attributes,
                operandsAndResultsTypes._2,
                operandsAndResultsTypes._1
              )
        ).tupled
      )
      ./
  )

  def CustomOperation[$: P](resNames: Seq[String]) = P(
    PrettyDialectReferenceName./.flatMapTry { (x: String, y: String) =>
      ctx.getOperation(s"${x}.${y}") match {
        case Some(y) =>
          Pass ~ y.parse(this)
        case None =>
          throw new Exception(
            s"Operation ${x}.${y} is not defined in any supported Dialect."
          )
      }
    }
  )

  def RegionList[$: P] =
    P("(" ~ Region.rep(sep = ",") ~ ")")

  /*≡==--==≡≡≡≡==--=≡≡*\
  ||      BLOCKS      ||
  \*≡==---==≡≡==---==≡*/

  // [x] - block           ::= block-label operation+
  // [x] - block-label     ::= block-id block-arg-list? `:`

  def createBlock(
      //            name    arguments       operations
      uncutBlock: (String, Seq[(String, Attribute)], ListType[Operation])
  ): Block = {
    val newBlock = new Block(
      operations = uncutBlock._3,
      arguments_types = ListType.from(uncutBlock._2.map(_._2))
    )
    for (op <- newBlock.operations) op.container_block = Some(newBlock)
    currentScope.defineBlock(uncutBlock._1, newBlock)
    currentScope.defineValues(uncutBlock._2.map(_._1) zip newBlock.arguments)
    return newBlock
  }

  def Block[$: P] =
    P(BlockLabel ~/ Operations(0)).mapTry(createBlock)

  def BlockLabel[$: P] = P(
    BlockId ~/ BlockArgList.orElse(Seq()) ~ ":"
  )

  /*≡==--==≡≡≡≡≡==--=≡≡*\
  ||      REGIONS      ||
  \*≡==---==≡≡≡==---==≡*/

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
          new Block(operations = parseResult._1, arguments_types = ListType())
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

  /*≡==--==≡≡≡≡==--=≡≡*\
  ||    ATTRIBUTES    ||
  \*≡==---==≡≡==---==≡*/

  // // Attribute Value Aliases
  // [x] - attribute-alias-def ::= `#` alias-name `=` attribute-value
  // [x] - attribute-alias ::= `#` alias-name

  def AttributeAliasDef[$: P] = P(
    "#" ~~ AliasName ~ "=" ~ AttributeValue
  )

  // [x] - value-id-and-type ::= value-id `:` type

  // // Non-empty list of names and types.
  // [x] - value-id-and-type-list ::= value-id-and-type (`,` value-id-and-type)*

  // [x] - block-arg-list ::= `(` value-id-and-type-list? `)`

  def ValueIdAndType[$: P] = P(ValueId ~ ":" ~ Type)

  def ValueIdAndTypeList[$: P] =
    P(ValueIdAndType.rep(sep = ",")).orElse(Seq())

  def BlockArgList[$: P] =
    P("(" ~/ ValueIdAndTypeList ~/ ")")

  // [x] dictionary-properties ::= `<` dictionary-attribute `>`
  // [x] dictionary-attribute  ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`

  /** Parses a properties dictionary, which synctatically simply is an attribute
    * dictionary wrapped in angle brackets.
    *
    * @return
    *   A properties dictionary parser.
    */
  def DictionaryProperties[$: P] = P(
    "<" ~ DictionaryAttribute ~ ">"
  )

  /** Parses an attributes dictionary.
    *
    * @return
    *   An attribute dictionary parser.
    */
  inline def DictionaryAttribute[$: P] = P(
    "{" ~ AttributeEntry
      .rep(sep = ",")
      .map(DictType[String, Attribute](_*)) ~ "}"
  )

  /** Parses an optional properties dictionary from the input.
    *
    * @return
    *   An optional dictionary of properties - empty if no dictionary is
    *   present.
    */
  inline def OptionalProperties[$: P]() =
    (DictionaryProperties).orElse(DictType.empty)

  /** Parses an optional attributes dictionary from the input.
    *
    * @return
    *   An optional dictionary of attributes - empty if no dictionary is
    *   present.
    */
  inline def OptionalAttributes[$: P] =
    (DictionaryAttribute).orElse(DictType.empty)

  /** Parses an optional attributes dictionary from the input, preceded by the
    * `attributes` keyword.
    *
    * @return
    *   An optional dictionary of attributes - empty if no keyword is present.
    */
  inline def OptionalKeywordAttributes[$: P] =
    ("attributes" ~/ DictionaryAttribute).orElse(DictType.empty)

  /*≡==--==≡≡≡≡==--=≡≡*\
  ||  PARSE FUNCTION  ||
  \*≡==---==≡≡==---==≡*/

  def parseThis[A, B](
      text: String,
      pattern: fastparse.P[_] => fastparse.P[B] = { (x: fastparse.P[_]) =>
        TopLevel(x)
      },
      verboseFailures: Boolean = false
  ): fastparse.Parsed[B] = {
    return parse(text, pattern, verboseFailures = verboseFailures)
  }

}

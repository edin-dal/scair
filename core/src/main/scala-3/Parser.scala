package scair

import fastparse.*
import fastparse.Implicits.Repeater
import fastparse.Parsed.Failure
import fastparse.internal.Util
import scair.clair.macros.DerivedOperationCompanion
import scair.dialects.builtin.ModuleOp
import scair.ir.*

import java.lang.Double.parseDouble
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

object Parser:

  import AttrParser.whitespace

  /*≡==--==≡≡≡≡==--=≡≡*\
  || COMMON FUNCTIONS ||
  \*≡==---==≡≡==---==≡*/

  // Custom function wrapper that allows to Escape out of a pattern
  // to carry out custom computation
  def E[$: P](action: => Unit) =
    action
    Pass(())

  def giveBack[A](a: A) =
    println(a)
    a

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
        try f(parsed)
        catch
          case e: Exception =>
            Fail(e.getMessage())
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
        try Pass(f(parsed))
        catch
          case e: Exception =>
            Fail(e.getMessage())
      )
    )

  /*≡==--==≡≡≡==--=≡≡*\
  ||      SCOPE      ||
  \*≡==---==≡==---==≡*/

  class Scope(
      var parentScope: Option[Scope] = None,
      var valueMap: mutable.Map[String, Value[Attribute]] =
        mutable.Map.empty[String, Value[Attribute]],
      var forwardValues: mutable.Set[String] = mutable.Set.empty[String],
      var blockMap: mutable.Map[String, Block] =
        mutable.Map.empty[String, Block],
      var forwardBlocks: mutable.Set[String] = mutable.Set.empty[String]
  ):

    def useValue(name: String, typ: Attribute): Value[Attribute] =
      valueMap.getOrElseUpdate(
        name, {
          forwardValues += name
          Value[Attribute](typ)
        }
      )

    def checkForwardedValues() =
      forwardValues.headOption match
        case Some(valueName) =>
          throw new Exception(s"Value %${valueName} not defined within Scope")
        case None => ()

    private def defineValue(name: String, typ: Attribute): Value[Attribute] =
      if valueMap.contains(name) then
        if !forwardValues.remove(name) then
          throw new Exception(
            s"Value cannot be defined twice within the same scope - %${name}"
          )
        valueMap(name)
      else
        val v = Value[Attribute](typ)
        valueMap(name) = v
        v

    inline def defineResult(name: String, typ: Attribute): Result[Attribute] =
      defineValue(name, typ)

    inline def defineBlockArgument(
        name: String,
        typ: Attribute
    ): BlockArgument[Attribute] =
      defineValue(name, typ)

    def checkForwardedBlocks() =
      forwardBlocks.headOption match
        case Some(blockName) =>
          throw new Exception(s"Successor ^$blockName not defined within Scope")
        case None => ()

    def defineBlock(
        blockName: String
    ): Block =
      if blockMap.contains(blockName) then
        if !forwardBlocks.remove(blockName) then
          throw new Exception(
            s"Block cannot be defined twice within the same scope - ^${blockName}"
          )
        blockMap(blockName)
      else
        val newBlock = new Block()
        blockMap(blockName) = newBlock
        newBlock

    def forwardBlock(
        blockName: String
    ): Block =
      blockMap.getOrElseUpdate(
        blockName, {
          forwardBlocks += blockName
          new Block()
        }
      )

    // child starts off from the parents context
    def createChild(): Scope =
      return new Scope(
        valueMap = valueMap.clone,
        parentScope = Some(this)
      )

    def switchWithParent: Scope =
      checkForwardedValues()
      checkForwardedBlocks()
      parentScope match
        case Some(x) =>
          x
        case None =>
          this

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

  inline val DecDigit = "0-9"
  inline val HexDigit = "0-9a-fA-F"

  inline def DecDigits[$: P] = CharsWhileIn(DecDigit)

  inline def HexDigits[$: P] = CharsWhileIn(HexDigit)

  inline val Letter = "a-zA-Z"
  inline val IdPunct = "$._\\-"

  def IntegerLiteral[$: P] = P(HexadecimalLiteral | DecimalLiteral)

  def DecimalLiteral[$: P] =
    P(("-" | "+").?.! ~ DecDigits.!).map((sign: String, literal: String) =>
      BigInt(sign + literal)
    )

  def HexadecimalLiteral[$: P] =
    P("0x" ~~ HexDigits.!).map((hex: String) => BigInt(hex, 16))

  private def parseFloatNum(float: (String, String)): Double =
    val number = parseDouble(float._1)
    val power = parseDouble(float._2)
    return number * pow(10, power)

  /** Parses a floating-point number from its string representation. NOTE: This
    * is only a float approximation, and in its current form should not be
    * trusted for precision sensitive applications.
    *
    * @return
    *   float: (String, String)
    */
  def FloatLiteral[$: P] = P(
    (CharIn("\\-\\+").? ~~ DecDigits ~~ "." ~~ DecDigits).!
      ~~ (CharIn("eE")
        ~~ (CharIn("\\-\\+").? ~~ DecDigits).!).orElse("0")
  ).map(parseFloatNum(_)) // substituted [0-9]* with [0-9]+

  inline def nonExcludedCharacter(c: Char): Boolean =
    c: @switch match
      case '"' | '\\' => false
      case _          => true

  inline def EscapedP[$: P] = P(
    ("\\" ~~ (
      "n" ~~ Pass('\n')
        | "t" ~~ Pass('\t')
        | "\\" ~~ Pass('\\')
        | "\"" ~~ Pass('\"')
        | CharIn("a-fA-F0-9")
          .repX(exactly = 2)
          .!
          .map(Integer.parseInt(_, 16).toChar)
    )).repX.map(chars => String(chars.toArray))
  )

  def StringLiteral[$: P] = P(
    "\"" ~~ (CharsWhile(nonExcludedCharacter).! ~~ EscapedP)
      .map(_ + _)
      .repX
      .map(_.mkString) ~~ "\""
  )

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

  def BareId[$: P] = P(
    CharIn(Letter + "_") ~~ CharsWhileIn(Letter + DecDigit + "_$.", min = 0)
  ).!

  def ValueId[$: P] = P("%" ~~ SuffixId)

  // Alias can't have dots in their names for ambiguity with dialect names.
  def AliasName[$: P] = P(
    CharIn(Letter + "_") ~~ (CharsWhileIn(
      Letter + DecDigit + "_$",
      min = 0
    )) ~~ !"."
  ).!

  def SuffixId[$: P] = P(
    DecimalLiteral | CharIn(Letter + IdPunct) ~~ CharsWhileIn(
      Letter + IdPunct + DecDigit,
      min = 0
    )
  ).!

  def SymbolRefId[$: P] = P("@" ~~ (SuffixId | StringLiteral))

  def ValueUse[$: P] =
    P(ValueId ~ ("#" ~~ DecimalLiteral).?).!.map(_.tail)

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

  def sequenceValues(value: (String, Option[BigInt])): Seq[String] =
    value match
      case (name, Some(totalNo)) =>
        (0 to (totalNo.toInt - 1)).map(no => s"$name#$no")
      case (name, None) => Seq(name)

  def OpResult[$: P] =
    P(ValueId ~ (":" ~ DecimalLiteral).?).map(sequenceValues)

  def TrailingLocation[$: P] = P(
    "loc" ~ "(" ~ "unknown" ~ ")"
  ) // definition taken from xdsl attribute_parser.py v-0.19.0 line 1106

  /*≡==--==≡≡≡≡==--=≡≡*\
  ||      BLOCKS      ||
  \*≡==---==≡≡==---==≡*/

  // [x] - block-id        ::= caret-id
  // [x] - caret-id        ::= `^` suffix-id

  def BlockId[$: P] = P(CaretId)

  def CaretId[$: P] = P("^" ~~/ SuffixId)

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
    CharPred(char => !excludedCharactersDTC.contains(char))
  )

  def DialectBareId[$: P] = P(
    CharIn(Letter + "_") ~~ CharsWhileIn(Letter + DecDigit + "_$", min = 0)
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

/*≡==--==≡≡≡≡≡≡≡≡==--=≡≡*\
||     PARSER CLASS     ||
\*≡==---==≡≡≡≡≡≡==---==≡*/

final class Parser(
    context: MLContext,
    inputPath: Option[String] = None,
    parsingDiagnostics: Boolean = false,
    allowUnregisteredDialect: Boolean = false,
    attributeAliases: mutable.Map[String, Attribute] = mutable.Map.empty,
    typeAliases: mutable.Map[String, Attribute] = mutable.Map.empty
) extends AttrParser(context, attributeAliases, typeAliases):

  import Parser.*
  import AttrParser.whitespace

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
      s"Parse error at ${inputPath.getOrElse("-")}:$line:$col:\n\n$input_line\n$indicator\n${traced.label}"

    if parsingDiagnostics then msg
    else
      Console.err.println(msg)
      sys.exit(1)

  var currentScope: Scope = new Scope()

  def enterLocalRegion =
    currentScope = currentScope.createChild()

  def enterParentRegion =
    currentScope = currentScope.switchWithParent

  /*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
  || TOP LEVEL PRODUCTION  ||
  \*≡==---==≡≡≡≡≡≡≡==---==≡*/

  // [x] toplevel := (operation | attribute-alias-def | type-alias-def)*
  // shortened definition TODO: finish...

  def TopLevel[$: P]: P[Operation] = P(
    Start ~ (OperationPat | AttributeAliasDef | TypeAliasDef)
      .rep()
      .map(_.flatMap(_ match
        case o: Operation => Seq(o)
        case _            => Seq())) ~ End
  ).map((toplevel: Seq[Operation]) =>
    toplevel.toList match
      case (head: ModuleOp) :: Nil => head
      case (head: DerivedOperationCompanion[ModuleOp]#UnstructuredOp) :: Nil =>
        head
      case _ =>
        val block = new Block(operations = toplevel)
        val region = Region(block)
        val moduleOp = ModuleOp(region)

        for op <- toplevel do op.container_block = Some(block)
        block.container_region = Some(region)
        region.container_operation = Some(moduleOp)

        moduleOp
  ) ~ E({
    currentScope.switchWithParent
  })

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
  def generateOperation[$: P](
      opName: String,
      resultsNames: Seq[String] = Seq(),
      operandsNames: Seq[String] = Seq(),
      successors: Seq[Block] = Seq(),
      properties: Map[String, Attribute] = Map(),
      regions: Seq[Region] = Seq(),
      attributes: Map[String, Attribute] = Map(),
      resultsTypes: Seq[Attribute] = Seq(),
      operandsTypes: Seq[Attribute] = Seq()
  ): P[Operation] =

    if operandsNames.length != operandsTypes.length then
      return Fail(
        s"Number of operands (${operandsNames.length}) does not match the number of the corresponding operand types (${operandsTypes.length}) in \"${opName}\"."
      )

    if resultsNames.length != resultsTypes.length then
      return Fail(
        s"Number of results (${resultsNames.length}) does not match the number of the corresponding result types (${resultsTypes.length}) in \"${opName}\"."
      )
    try
      val operands = operandsNames zip operandsTypes map currentScope.useValue
      val results = resultsNames zip resultsTypes map currentScope.defineResult

      ctx.getOpCompanion(opName, allowUnregisteredDialect) match
        case Right(companion) =>
          Pass(
            companion(
              operands = operands,
              successors = successors,
              properties = properties,
              results = results,
              attributes = DictType.from(attributes),
              regions = regions
            )
          )
        case Left(error) => Fail(error)

    catch case e: Exception => return Fail(e.getMessage())

  def Operations[$: P](
      at_least_this_many: Int = 0
  ): P[Seq[Operation]] =
    P(OperationPat.rep(at_least_this_many))

  def OperationPat[$: P]: P[Operation] = P(
    OpResultList.orElse(Seq())./.flatMap(Op(_)) ~/ TrailingLocation.?
  )./

  def Op[$: P](resNames: Seq[String]) = P(
    GenericOperation(resNames) | CustomOperation(resNames)
  ).map(op =>
    for region <- op.regions do region.container_operation = Some(op)
    op
  )

  def GenericOperationName[$: P]: P[OperationCompanion[?]] =
    StringLiteral.flatMap(ctx.getOpCompanion(_, allowUnregisteredDialect) match
      case Right(companion) => Pass(companion)
      case Left(error)      => Fail(error))

  def GenericOperandsTypesRec[$: P](using
      expected: Int
  )(operandsNames: Seq[String], parsed: Int = 1): P[Seq[Value[Attribute]]] =
    operandsNames match
      case head :: Nil =>
        Type.mapTry(currentScope.useValue(head, _)).map(Seq(_))
      case head :: tail =>
        (Type.mapTry(currentScope.useValue(head, _)) ~ ",".opaque(
          f"Number of operands ($expected) does not match the number of the corresponding operand types ($parsed)."
        ) ~ GenericOperandsTypesRec(
          tail,
          parsed + 1
        )).map(_ +: _)
      case Nil => Pass(Seq())

  def GenericOperandsTypes[$: P](
      operandsNames: Seq[String]
  ): P[Seq[Value[Attribute]]] =
    "(" ~ GenericOperandsTypesRec(using operandsNames.length)(
      operandsNames
    ) ~ ")"

  def GenericResultsTypesRec[$: P](using
      expected: Int
  )(
      resultsNames: Seq[String],
      parsed: Int = 1
  ): P[Seq[Result[Attribute]]] =
    resultsNames match
      case head :: Nil =>
        Type
          .mapTry(
            currentScope.defineResult(head, _)
          )
          .map(Seq(_))
      case head :: tail =>
        (Type.mapTry(
          currentScope.defineResult(head, _)
        ) ~ ",".opaque(
          f"Number of results ($expected) does not match the number of the corresponding result types ($parsed)."
        ) ~ GenericResultsTypesRec(tail, parsed + 1)).map(_ +: _)
      case Nil => Pass(Seq())

  def GenericResultsTypes[$: P](
      resultsNames: Seq[String]
  ): P[Seq[Result[Attribute]]] =
    ("("./ ~ GenericResultsTypesRec(using resultsNames.length)(
      resultsNames
    ) ~ ")") | Pass(())
      .filter(_ => resultsNames.length == 1)
      .flatMap(_ =>
        GenericResultsTypesRec(using resultsNames.length)(resultsNames)
      )

  def GenericOperationType[$: P](
      resultsNames: Seq[String],
      operandsNames: Seq[String]
  ) =
    P(
      GenericOperandsTypes(operandsNames)
        ~ "->" ~/ GenericResultsTypes(resultsNames)
    )

  def GenericOperation[$: P](resultsNames: Seq[String]): P[Operation] = P(
    GenericOperationName
      .flatMapX((opCompanion: OperationCompanion[?]) =>
        "(" ~ ValueUseList
          .orElse(Seq())
          .flatMap((operandsNames: Seq[String]) =>
            (")"
              ~/ SuccessorList.orElse(Seq())
              ~/ properties.orElse(Map.empty)
              ~/ RegionList.orElse(Seq())
              ~/ OptionalAttributes ~/ ":" ~/ GenericOperationType(
                resultsNames,
                operandsNames
              ))
              .flatMap(
                (
                    successors: Seq[Block],
                    properties: Map[String, Attribute],
                    regions: Seq[Region],
                    attributes: Map[String, Attribute],
                    operandsAndResults: (
                        Seq[Value[Attribute]],
                        Seq[Result[Attribute]]
                    )
                ) =>
                  val (operands, results) = operandsAndResults
                  Pass(
                    opCompanion(
                      operands,
                      successors,
                      results,
                      regions,
                      properties,
                      attributes.to(DictType)
                    )
                  )
              )
          )
      )
      ./
  )

  def CustomOperation[$: P](resNames: Seq[String]) = P(
    PrettyDialectReferenceName./.flatMapTry { (x: String, y: String) =>
      ctx.getOpCompanion(s"${x}.${y}") match
        case Right(companion) =>
          Pass ~ companion.parse(this, resNames)
        case Left(_) =>
          Fail(
            s"Operation ${x}.${y} is not defined in any supported Dialect."
          )
    }
  )

  def RegionList[$: P] =
    P("(" ~ RegionP().rep(sep = ",") ~ ")")

  /*≡==--==≡≡≡≡==--=≡≡*\
  ||      BLOCKS      ||
  \*≡==---==≡≡==---==≡*/

  // [x] - block           ::= block-label operation+
  // [x] - block-label     ::= block-id block-arg-list? `:`

  def populateBlockOps(
      block: Block,
      ops: Seq[Operation]
  ): Block =
    block.operations ++= ops
    ops.foreach(_.container_block = Some(block))
    block

  def populateBlockArgs(
      block: Block,
      args: Seq[(String, Attribute)]
  ) =
    block.arguments ++= args map currentScope.defineResult
    block.arguments.foreach(_.owner = Some(block))
    block

  def BlockBody[$: P](block: Block) =
    Operations(0).mapTry(populateBlockOps(block, _))

  def Block[$: P] =
    P(BlockLabel.flatMap(BlockBody))

  def BlockLabel[$: P] = P(
    (BlockId.mapTry(currentScope.defineBlock) ~/ BlockArgList
      .orElse(Seq())).map(populateBlockArgs) ~ ":"
  )

  def SuccessorList[$: P] = P("[" ~ Successor.rep(sep = ",") ~ "]")

  def Successor[$: P] =
    P(CaretId).map(currentScope.forwardBlock) // possibly shortened version

  /*≡==--==≡≡≡≡≡==--=≡≡*\
  ||      REGIONS      ||
  \*≡==---==≡≡≡==---==≡*/

  // [x] - region        ::= `{` entry-block? block* `}`
  // [x] - entry-block   ::= operation+
  //                    |
  //                    |  rewritten as
  //                   \/
  // [x] - region        ::= `{` operation* block* `}`

  def defineRegion(
      parseResult: (Seq[Operation], Seq[Block])
  ): Region =
    return parseResult._1.length match
      case 0 =>
        val region = Region(blocks = parseResult._2)
        for block <- region.blocks do block.container_region = Some(region)
        region
      case _ =>
        val startblock =
          new Block(operations = parseResult._1, arguments_types = ListType())
        val region = Region(blocks = startblock +: parseResult._2)
        for block <- region.blocks do block.container_region = Some(region)
        region

  // EntryBlock might break - take out if it does...
  def RegionP[$: P](entryArgs: Seq[(String, Attribute)] = Seq()) = P(
    "{" ~/ E(
      { enterLocalRegion }
    ) ~/ (BlockBody(populateBlockArgs(new Block(), entryArgs)) ~/ Block.rep)
      .map((entry: Block, blocks: Seq[Block]) =>
        if entry.operations.isEmpty && entry.arguments.isEmpty then blocks
        else entry +: blocks
      ) ~/ "}"
  ).map(Region(_)) ~/ E(
    { enterParentRegion }
  )

  // def EntryBlock[$: P] = P(OperationPat.rep(1))

  // // Type aliases
  // [x] type-alias-def ::= `!` alias-name `=` type
  // [x] type-alias ::= `!` alias-name

  def TypeAliasDef[$: P] = P(
    "!" ~~ AliasName ~ "=" ~ Type
  )./.map((name: String, value: Attribute) =>
    typeAliases.get(name) match
      case Some(t) =>
        throw new Exception(
          s"""Type alias "$name" already defined as $t."""
        )
      case None =>
        typeAliases(name) = value
  )

  /*≡==--==≡≡≡≡==--=≡≡*\
  ||    ATTRIBUTES    ||
  \*≡==---==≡≡==---==≡*/

  // // Attribute Value Aliases
  // [x] - attribute-alias-def ::= `#` alias-name `=` attribute-value
  // [x] - attribute-alias ::= `#` alias-name

  def AttributeAliasDef[$: P] = P(
    "#" ~~ AliasName ~ "=" ~ Attribute
  )./.map((name: String, value: Attribute) =>
    attributeAliases.get(name) match
      case Some(a) =>
        throw new Exception(
          s"""Attribute alias "$name" already defined as $a."""
        )
      case None =>
        attributeAliases(name) = value
  )

  // [x] - value-id-and-type ::= value-id `:` type

  // // Non-empty list of names and types.
  // [x] - value-id-and-type-list ::= value-id-and-type (`,` value-id-and-type)*

  // [x] - block-arg-list ::= `(` value-id-and-type-list? `)`

  def ValueIdAndType[$: P] = P(ValueId ~ ":" ~ Type)

  def ValueIdAndTypeList[$: P] =
    P(ValueIdAndType.rep(sep = ",")).orElse(Seq())

  def BlockArgList[$: P] =
    P("(" ~ ValueIdAndTypeList ~ ")")

  // [x] dictionary-properties ::= `<` dictionary-attribute `>`
  // [x] dictionary-attribute  ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`

  /** Parses a properties dictionary, which synctatically simply is an attribute
    * dictionary wrapped in angle brackets.
    *
    * @return
    *   A properties dictionary parser.
    */
  def properties[$: P] = P(
    "<" ~ DictionaryAttribute ~ ">"
  )

  /** Parses an attributes dictionary.
    *
    * @return
    *   An attribute dictionary parser.
    */
  inline def DictionaryAttribute[$: P] = P(
    DictionaryAttributeP.map(_.entries)
  )

  /** Parses an optional properties dictionary from the input.
    *
    * @return
    *   An optional dictionary of properties - empty if no dictionary is
    *   present.
    */
  inline def OptionalProperties[$: P]() =
    (properties).orElse(DictType.empty)

  /** Parses an optional attributes dictionary from the input.
    *
    * @return
    *   An optional dictionary of attributes - empty if no dictionary is
    *   present.
    */
  inline def OptionalAttributes[$: P] =
    (DictionaryAttribute).orElse(Map.empty)

  /** Parses an optional attributes dictionary from the input, preceded by the
    * `attributes` keyword.
    *
    * @return
    *   An optional dictionary of attributes - empty if no keyword is present.
    */
  inline def OptionalKeywordAttributes[$: P] =
    ("attributes" ~/ DictionaryAttribute).orElse(Map.empty)

  /*≡==--==≡≡≡≡==--=≡≡*\
  ||  PARSE FUNCTION  ||
  \*≡==---==≡≡==---==≡*/

  def parseThis[A, B](
      text: String,
      pattern: fastparse.P[?] => fastparse.P[B] = { (x: fastparse.P[?]) =>
        TopLevel(using x)
      },
      verboseFailures: Boolean = false
  ): fastparse.Parsed[B] =
    return parse(text, pattern, verboseFailures = verboseFailures)

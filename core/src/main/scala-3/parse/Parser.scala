package scair.parse

import fastparse.*
import fastparse.Implicits.Repeater
import fastparse.Parsed.Failure
import fastparse.internal.Util
import scair.MLContext
import scair.clair.macros.DerivedOperationCompanion
import scair.dialects.builtin.ModuleOp
import scair.ir.*

import scala.annotation.tailrec
import scala.collection.mutable

// ██████╗░ ░█████╗░ ██████╗░ ░██████╗ ███████╗ ██████╗░
// ██╔══██╗ ██╔══██╗ ██╔══██╗ ██╔════╝ ██╔════╝ ██╔══██╗
// ██████╔╝ ███████║ ██████╔╝ ╚█████╗░ █████╗░░ ██████╔╝
// ██╔═══╝░ ██╔══██║ ██╔══██╗ ░╚═══██╗ ██╔══╝░░ ██╔══██╗
// ██║░░░░░ ██║░░██║ ██║░░██║ ██████╔╝ ███████╗ ██║░░██║
// ╚═╝░░░░░ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ╚═════╝░ ╚══════╝ ╚═╝░░╚═╝

/*≡==--==≡≡≡≡==--=≡≡*\
|| COMMON FUNCTIONS ||
\*≡==---==≡≡==---==≡*/

extension [T](inline p: P[T])

  /** Make the parser optional, parsing defaults if otherwise failing.
    *
    * @todo:
    *   Figure out dark implicit magic to figure out magically that the default
    *   default is "T()".
    *
    * @param default
    *   The default value to use if the parser fails.
    * @return
    *   An optional parser, defaulting to default.
    */
  inline def orElse[$: P](inline default: T): P[T] = P(
    p | Pass(default)
  )

  /** Like fastparse's flatMapX but capturing exceptions as standard parse
    * errors.
    *
    * @note
    *   flatMapX because it often yields more natural error positions.
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
          Console.err.print(
            "WARNING: Caught an exception in parsing; this is deprecated, use fastparse's Fail instead.\n"
          )
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
          Console.err.print(
            "WARNING: Caught an exception in parsing; this is deprecated, use fastparse's Fail instead.\n"
          )
          Fail(e.getMessage())
    )
  )

/*≡==--==≡≡≡==--=≡≡*\
||      SCOPE      ||
\*≡==---==≡==---==≡*/

private final class Scope(
    var valueMap: mutable.Map[String, Value[Attribute]] =
      mutable.Map.empty[String, Value[Attribute]],
    var forwardValues: mutable.Set[String] = mutable.Set.empty[String],
    var blockMap: mutable.Map[String, Block] = mutable.Map.empty[String, Block],
    var forwardBlocks: mutable.Set[String] = mutable.Set.empty[String]
):

  def allBlocksAndValuesDefined[$: P] =
    forwardValues.headOption match
      case Some(valueName) =>
        Fail(s"Value %${valueName} not defined within Scope")
      case None =>
        forwardBlocks.headOption match
          case Some(blockName) =>
            Fail(s"Successor ^$blockName not defined within Scope")
          case None => Pass

  def defineValue[$: P](
      name: String,
      typ: Attribute
  ): P[Value[Attribute]] =
    if valueMap.contains(name) then
      if !forwardValues.remove(name) then
        Fail(
          s"Value cannot be defined twice within the same scope - %${name}"
        )
      Pass(valueMap(name))
    else
      val v = Value[Attribute](typ)
      valueMap(name) = v
      Pass(v)

  inline def defineBlockArgument[$: P](
      name: String,
      typ: Attribute
  ): P[BlockArgument[Attribute]] =
    defineValue(name, typ).map(_.asInstanceOf[BlockArgument[Attribute]])

  def defineBlock[$: P](
      blockName: String
  ): P[Block] =
    if blockMap.contains(blockName) then
      if !forwardBlocks.remove(blockName) then
        Fail(
          f"Block cannot be defined twice within the same scope - ^${blockName}"
        )
      else Pass(blockMap(blockName))
    else
      val newBlock = new Block()
      blockMap(blockName) = newBlock
      Pass(newBlock)

  def forwardBlock(
      blockName: String
  ): Block =
    blockMap.getOrElseUpdate(
      blockName, {
        forwardBlocks += blockName
        new Block()
      }
    )

/*≡==--==≡≡≡≡==--=≡≡*\
||    OPERATIONS    ||
\*≡==---==≡≡==---==≡*/

// [x] op-result-list        ::= op-result (`,` op-result)* `=`
// [x] op-result             ::= value-id (`:` integer-literal)?
// [x] successor-list        ::= `[` successor (`,` successor)* `]`
// [x] successor             ::= caret-id (`:` block-arg-list)?
// [x] trailing-location     ::= `loc` `(` location `)`

private inline def OpResultList[$: P] =
  (OpResult.rep(1, sep = ",") ~ "=").orElse(Seq()).map(_.flatten)

private inline def sequenceValues(
    value: (String, Option[BigInt])
): Seq[String] =
  value match
    case (name, Some(totalNo)) =>
      (0 to (totalNo.toInt - 1)).map(no => s"$name#$no")
    case (name, None) => Seq(name)

private inline def OpResult[$: P] =
  (ValueId ~ (":" ~ DecimalLiteral).?).map(sequenceValues)

private inline def TrailingLocation[$: P] = "loc" ~ "(" ~ "unknown" ~ ")"

/*≡==--==≡≡≡≡≡≡≡≡==--=≡≡*\
||     PARSER CLASS     ||
\*≡==---==≡≡≡≡≡≡==---==≡*/

final class Parser(
    private[parse] context: MLContext,
    private[parse] final val inputPath: Option[String] = None,
    private[parse] final val parsingDiagnostics: Boolean = false,
    private[parse] final val allowUnregisteredDialect: Boolean = false,
    private[parse] attributeAliases: mutable.Map[String, Attribute] =
      mutable.Map.empty,
    private[parse] typeAliases: mutable.Map[String, Attribute] =
      mutable.Map.empty,
    private[parse] val scopes: mutable.Stack[Scope] = mutable.Stack(new Scope())
) extends AttrParser(context, attributeAliases, typeAliases):

  private[parse] inline def enterRegion[$: P] =
    scopes.push(new Scope())
    Pass

  private[parse] inline def exitRegion[$: P] =
    scopes.pop.allBlocksAndValuesDefined

  inline def parse[T](
      inline input: ParserInputSource,
      inline parser: P[?] => P[T] = TopLevelP(using _, this),
      inline verboseFailures: Boolean = false,
      inline startIndex: Int = 0,
      inline instrument: fastparse.internal.Instrument = null
  ) =
    fastparse.parse(
      input,
      parser,
      verboseFailures,
      startIndex,
      instrument
    )

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

    given Parser = this

    if operandsNames.length != operandsTypes.length then
      return Fail(
        s"Number of operands (${operandsNames.length}) does not match the number of the corresponding operand types (${operandsTypes.length}) in \"${opName}\"."
      )

    if resultsNames.length != resultsTypes.length then
      return Fail(
        s"Number of results (${resultsNames.length}) does not match the number of the corresponding result types (${resultsTypes.length}) in \"${opName}\"."
      )

    (operandsNames zip operandsTypes)
      .foldLeft(
        Pass(Seq.empty[Value[Attribute]])
      )((l: P[Seq[Value[Attribute]]], r: (String, Attribute)) =>
        (l ~ operand(r._1, r._2)).map(_ :+ _)
      )
      .flatMap(operands =>
        (resultsNames zip resultsTypes)
          .foldLeft(
            Pass(Seq.empty[Result[Attribute]])
          )((l: P[Seq[Result[Attribute]]], r: (String, Attribute)) =>
            (l ~ result(r._1, r._2)).map(_ :+ _)
          )
          .flatMap(results =>
            context.getOpCompanion(opName, allowUnregisteredDialect) match
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
          )
      )

  def error(failure: Failure, lineOffset: Int = 0) =
    // .trace() below reparses from the start with more bookkeeping to provide helpful
    // context for the error message.
    // We do this very non-functional bookkeeping in currentScope ourselves, which
    // is disfunctional with this behaviour; it then reparses everything with the state
    // it already had at the time of the catched error!
    // This is a workaround to get the error message with the correct state.
    // TODO: More functional and fastparse-compatible state handling!
    scopes.popAll
    scopes.push(new Scope())

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
      s"Parse error at ${inputPath.getOrElse("-")}:${line + lineOffset}:$col:\n\n$input_line\n$indicator\n${traced.label}"

    if parsingDiagnostics then msg
    else
      Console.err.println(msg)
      sys.exit(1)

def operand[$: P](name: String, typ: Attribute)(using
    p: Parser
): P[Value[Attribute]] =
  p.scopes.collectFirst {
    case scope if scope.valueMap.contains(name) =>
      scope.valueMap(name)
  } match
    case Some(value) if value.typ == typ => Pass(value)
    case Some(value)                     =>
      Fail(
        s"Value %${name} defined with type ${value.typ}, but used with type ${typ}."
      )
    case None =>
      val forwardValue = Value[Attribute](typ)
      p.scopes.top.valueMap(name) = forwardValue
      p.scopes.top.forwardValues += name
      Pass(forwardValue)

def result[$: P](
    name: String,
    typ: Attribute
)(using p: Parser): P[Result[Attribute]] =
  P(
    p.scopes.top.defineValue(name, typ).map(_.asInstanceOf[Result[Attribute]])
  )

/*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
|| TOP LEVEL PRODUCTION  ||
\*≡==---==≡≡≡≡≡≡≡==---==≡*/

// [x] toplevel := (operation | attribute-alias-def | type-alias-def)*
// shortened definition TODO: finish...

def TopLevelP[$: P](using p: Parser): P[Operation] = P(
  Start ~ (OperationP | AttributeAliasDef | TypeAliasDef).rep.map(
    _.collect { case o: Operation =>
      o
    }
  ) ~/ p.exitRegion ~ End
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

def OperationP[$: P](using Parser): P[Operation] = P(
  OpResultList./.flatMap(resNames =>
    GenericOperation(resNames) | CustomOperation(resNames)
  ) ~/ TrailingLocation.?
)./

private def GenericOperandsTypesRec[$: P](using
    expected: Int,
    p: Parser
)(operandsNames: Seq[String], parsed: Int = 1): P[Seq[Value[Attribute]]] =
  operandsNames match
    case head :: Nil =>
      TypeP
        .opaque(
          f"Number of operands ($expected) does not match the number of the corresponding operand types (${parsed - 1})."
        )
        .flatMap(operand(head, _))
        .map(Seq(_))
    case head :: tail =>
      (TypeP.flatMap(operand(head, _)) ~ ",".opaque(
        f"Number of operands ($expected) does not match the number of the corresponding operand types ($parsed)."
      ) ~ GenericOperandsTypesRec(
        tail,
        parsed + 1
      )).map(_ +: _)
    case Nil => Pass(Seq())

private inline def GenericOperandsTypes[$: P](
    operandsNames: Seq[String]
)(using Parser): P[Seq[Value[Attribute]]] =
  "(" ~ GenericOperandsTypesRec(using operandsNames.length)(
    operandsNames
  ) ~ ")"

private def GenericResultsTypesRec[$: P](using
    expected: Int,
    p: Parser
)(
    resultsNames: Seq[String],
    parsed: Int = 1
): P[Seq[Result[Attribute]]] =
  resultsNames match
    case head :: Nil =>
      TypeP
        .opaque(
          f"Number of results ($expected) does not match the number of the corresponding result types (${parsed - 1})."
        )
        .flatMap(
          result(head, _)
        )
        .map(Seq(_))
    case head :: tail =>
      (TypeP.flatMap(
        result(head, _)
      ) ~ ",".opaque(
        f"Number of results ($expected) does not match the number of the corresponding result types ($parsed)."
      ) ~ GenericResultsTypesRec(tail, parsed + 1)).map(_ +: _)
    case Nil => Pass(Seq())

private inline def GenericResultsTypes[$: P](
    resultsNames: Seq[String]
)(using Parser): P[Seq[Result[Attribute]]] =
  ("(" ~/ GenericResultsTypesRec(using resultsNames.length)(
    resultsNames
  ).flatMap(resultsTypes =>
    ")".opaque(
      f"Number of results (${resultsNames.length}) does not match the number of the corresponding result types."
    ) ~ Pass(resultsTypes)
  )) | Pass(())
    .filter(_ => resultsNames.length == 1)
    .flatMap(_ =>
      GenericResultsTypesRec(using resultsNames.length)(resultsNames)
    )

private inline def GenericOperationName[$: P](using
    p: Parser
): P[OperationCompanion[?]] =
  StringLiteral.flatMap(
    p.context.getOpCompanion(_, p.allowUnregisteredDialect) match
      case Right(companion) => Pass(companion)
      case Left(error)      => Fail(error)
  )

private inline def GenericOperation[$: P](
    resultsNames: Seq[String]
)(using Parser): P[Operation] =
  GenericOperationName
    .flatMapX((opCompanion: OperationCompanion[?]) =>
      "(" ~ OperandNames
        .orElse(Seq())
        .flatMap((operandsNames: Seq[String]) =>
          (")"
            ~/ SuccessorList.orElse(Seq())
            ~/ properties.orElse(Map.empty)
            ~/ RegionList.orElse(Seq())
            ~/ OptionalAttributes ~/ ":" ~/ GenericOperandsTypes(
              operandsNames
            )
            ~ "->" ~/ GenericResultsTypes(resultsNames))
            .map(
              (
                  successors: Seq[Block],
                  properties: Map[String, Attribute],
                  regions: Seq[Region],
                  attributes: Map[String, Attribute],
                  operands: Seq[Value[Attribute]],
                  results: Seq[Result[Attribute]]
              ) =>
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

private inline def CustomOperation[$: P](
    resNames: Seq[String]
)(using p: Parser) =
  PrettyDialectReferenceName./.flatMapTry { (x: String, y: String) =>
    p.context.getOpCompanion(s"${x}.${y}") match
      case Right(companion) =>
        Pass ~ companion.parse(resNames)
      case Left(_) =>
        Fail(
          s"Operation ${x}.${y} is not defined in any supported Dialect."
        )
  }

private inline def RegionList[$: P](using Parser) =
  "(" ~ RegionP().rep(sep = ",") ~ ")"

// // Type aliases
// [x] type-alias-def ::= `!` alias-name `=` type
// [x] type-alias ::= `!` alias-name

private inline def TypeAliasDef[$: P](using p: Parser) =
  ("!" ~~ AliasName ~ "=" ~ TypeP).flatMap((name: String, value: Attribute) =>
    p.typeAliases.get(name) match
      case Some(t) =>
        Fail(
          s"""Type alias "$name" already defined as $t."""
        )
      case None =>
        p.typeAliases(name) = value
        Pass
  )

/*≡==--==≡≡≡≡==--=≡≡*\
||    ATTRIBUTES    ||
\*≡==---==≡≡==---==≡*/

// // Attribute Value Aliases
// [x] - attribute-alias-def ::= `#` alias-name `=` attribute-value
// [x] - attribute-alias ::= `#` alias-name

private inline def AttributeAliasDef[$: P](using p: Parser) = (
  "#" ~~ AliasName ~ "=" ~ AttributeP
).flatMap((name: String, value: Attribute) =>
  p.attributeAliases.get(name) match
    case Some(a) =>
      Fail(
        s"""Attribute alias "$name" already defined as $a."""
      )
    case None =>
      p.attributeAliases(name) = value
      Pass
)

/*≡==--==≡≡≡≡==--=≡≡*\
||      BLOCKS      ||
\*≡==---==≡≡==---==≡*/

// [x] - block           ::= block-label operation+
// [x] - block-label     ::= block-id block-arg-list? `:`

private inline def populateBlockOps(
    block: Block,
    ops: Seq[Operation]
)(using p: Parser): Block =
  block.operations ++= ops
  ops.foreach(_.container_block = Some(block))
  block

private inline def populateBlockArgs[$: P](
    block: Block,
    args: Seq[(String, Attribute)]
)(using p: Parser) =
  args
    .foldLeft(Pass(Seq.empty[BlockArgument[Attribute]]))((l, r) =>
      (l ~ p.scopes.top.defineBlockArgument(r._1, r._2)).map(_ :+ _)
    )
    .map(args =>
      block.arguments ++= args
      block.arguments.foreach(_.owner = Some(block))
      block
    )

private inline def BlockBody[$: P](block: Block)(using Parser) =
  OperationP.rep.mapTry(populateBlockOps(block, _))

def BlockP[$: P](using Parser) = P(
  P(BlockLabel.flatMap(BlockBody))
)

private inline def BlockLabel[$: P](using p: Parser) =
  (BlockId.flatMap(p.scopes.top.defineBlock) ~ (BlockArgList
    .orElse(Seq()))).flatMap(populateBlockArgs) ~ ":"

inline def SuccessorList[$: P](using Parser) = P(
  "[" ~ Successor.rep(sep = ",") ~ "]"
)

inline def Successor[$: P](using p: Parser) = P(
  P(CaretId).map(p.scopes.top.forwardBlock)
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

def RegionP[$: P](
    entryArgs: Seq[(String, Attribute)] = Seq()
)(using p: Parser) = P(
  "{" ~/ p.enterRegion ~/ (populateBlockArgs(new Block(), entryArgs)
    .flatMap(BlockBody) ~/ BlockP.rep)
    .map((entry: Block, blocks: Seq[Block]) =>
      if entry.operations.isEmpty && entry.arguments.isEmpty then blocks
      else entry +: blocks
    ) ~/ "}" ~/ p.exitRegion
).map(Region(_))

// [x] - value-id-and-type ::= value-id `:` type

// // Non-empty list of names and types.
// [x] - value-id-and-type-list ::= value-id-and-type (`,` value-id-and-type)*

// [x] - block-arg-list ::= `(` value-id-and-type-list? `)`

inline def ValueIdAndType[$: P](using Parser) = P(ValueId ~ ":" ~ TypeP)

private inline def ValueIdAndTypeList[$: P](using Parser) =
  P(ValueIdAndType.rep(sep = ",")).orElse(Seq())

private inline def BlockArgList[$: P](using Parser) =
  P(
    "(" ~ ValueIdAndType
      .rep(sep = ",") ~ ")"
  )

// [x] dictionary-properties ::= `<` dictionary-attribute `>`
// [x] dictionary-attribute  ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`

/** Parses a properties dictionary, which synctatically simply is an attribute
  * dictionary wrapped in angle brackets.
  *
  * @return
  *   A properties dictionary parser.
  */
inline def properties[$: P](using Parser) = P(
  "<" ~ DictionaryAttribute ~ ">"
)

/** Parses an attributes dictionary.
  *
  * @return
  *   An attribute dictionary parser.
  */
inline def DictionaryAttribute[$: P](using Parser) = P(
  DictionaryAttributeP.map(_.entries)
)

/** Parses an optional properties dictionary from the input.
  *
  * @return
  *   An optional dictionary of properties - empty if no dictionary is present.
  */
inline def OptionalProperties[$: P](using Parser) =
  (properties).orElse(DictType.empty)

/** Parses an optional attributes dictionary from the input.
  *
  * @return
  *   An optional dictionary of attributes - empty if no dictionary is present.
  */
inline def OptionalAttributes[$: P](using Parser) =
  (DictionaryAttribute).orElse(Map.empty)

/** Parses an optional attributes dictionary from the input, preceded by the
  * `attributes` keyword.
  *
  * @return
  *   An optional dictionary of attributes - empty if no keyword is present.
  */
inline def OptionalKeywordAttributes[$: P](using Parser) =
  ("attributes" ~/ DictionaryAttribute).orElse(Map.empty)

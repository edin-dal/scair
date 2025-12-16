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
  inline def orElse[$: P](inline default: => T): P[T] = P(
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

  // Replacement for fastparse's .opaque, with a by-name message, so as not to build it in the happy case.
  // TODO: Should that be contributed to fastparse's .opauqe or does it have a reason not to be?
  inline def explain[$: P](inline msg: => String): P[T] =
    p | Fail(msg)

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

// See uses; enables .rep to concatenate parsed sequences
// TODO: Expose as nicer helper, but could'nt get it just right for now
def concatRepeater[T] = new Repeater[Seq[T], Seq[T]]:
  type Acc = mutable.Buffer[T]
  def initial = mutable.Buffer.empty[T]
  def accumulate(t: Seq[T], acc: mutable.Buffer[T]) = acc ++= t
  def result(acc: mutable.Buffer[T]) = acc.toSeq

/*≡==--==≡≡≡==--=≡≡*\
||      SCOPE      ||
\*≡==---==≡==---==≡*/

private final class Scope(
    var valueMap: mutable.Map[String, Value[Attribute]] = mutable.Map
      .empty[String, Value[Attribute]],
    var forwardValues: mutable.Set[String] = mutable.Set.empty[String],
    var blockMap: mutable.Map[String, Block] = mutable.Map.empty[String, Block],
    var forwardBlocks: mutable.Set[String] = mutable.Set.empty[String],
):

  inline def allBlocksAndValuesDefinedP[$: P] =
    forwardValues.headOption match
      case Some(valueName) =>
        Fail(s"Value %$valueName not defined within Scope")
      case None =>
        forwardBlocks.headOption match
          case Some(blockName) =>
            Fail(s"Successor ^$blockName not defined within Scope")
          case None => Pass

  inline def defineValueP[$: P](
      name: String,
      typ: Attribute,
  ): P[Value[Attribute]] =
    if valueMap.contains(name) then
      if !forwardValues.remove(name) then
        Fail(
          s"Value cannot be defined twice within the same scope - %$name"
        )
      Pass(valueMap(name))
    else
      val v = Value[Attribute](typ)
      valueMap(name) = v
      Pass(v)

  def defineBlockArgumentP[$: P](
      name: String,
      typ: Attribute,
  ): P[BlockArgument[Attribute]] =
    defineValueP(name, typ).map(_.asInstanceOf[BlockArgument[Attribute]])

  inline def defineBlockP[$: P](
      blockName: String
  ): P[Block] =
    if blockMap.contains(blockName) then
      if !forwardBlocks.remove(blockName) then
        Fail(
          f"Block cannot be defined twice within the same scope - ^$blockName"
        )
      else Pass(blockMap(blockName))
    else
      val newBlock = new Block()
      blockMap(blockName) = newBlock
      Pass(newBlock)

  inline def forwardBlock(
      blockName: String
  ): Block =
    blockMap.getOrElseUpdate(
      blockName, {
        forwardBlocks += blockName
        new Block()
      },
    )

/*≡==--==≡≡≡≡==--=≡≡*\
||    OPERATIONS    ||
\*≡==---==≡≡==---==≡*/

// [x] op-result-list        ::= op-result (`,` op-result)* `=`
// [x] op-result             ::= value-id (`:` integer-literal)?
// [x] successor-list        ::= `[` successor (`,` successor)* `]`
// [x] successor             ::= caret-id (`:` block-arg-list)?
// [x] trailing-location     ::= `loc` `(` location `)`

private def opResultListP[$: P] =
  (opResultP.rep(1, sep = ",")(using concatRepeater[String]) ~ "=")
    .orElse(Seq())

private inline def sequenceValues(
    name: String,
    no: BigInt,
): Seq[String] = (0 to (no.toInt - 1)).map(no => s"$name#$no")

private inline def opResultP[$: P] = (valueIdP.flatMapX(name =>
  (":" ~~ decDigitsP.!.map(d => sequenceValues(name, d.toInt)))
    .orElse(Seq(name))
))

private def trailingLocationP[$: P] = "loc" ~ "(" ~ "unknown" ~ ")"

/*≡==--==≡≡≡≡≡≡≡≡==--=≡≡*\
||     PARSER CLASS     ||
\*≡==---==≡≡≡≡≡≡==---==≡*/

final class Parser(
    private[parse] final val context: MLContext,
    private[parse] final val inputPath: Option[String] = None,
    private[parse] final val parsingDiagnostics: Boolean = false,
    private[parse] final val allowUnregisteredDialect: Boolean = false,
    private[parse] final val attributeAliases: mutable.Map[String, Attribute] =
      mutable.Map.empty,
    private[parse] final val typeAliases: mutable.Map[String, Attribute] =
      mutable.Map.empty,
    private[parse] final val scopes: mutable.Stack[Scope] = mutable
      .Stack(new Scope()),
):

  private[parse] def enterRegionP[$: P] =
    scopes.push(new Scope())
    Pass

  private[parse] def exitRegionP[$: P] =
    scopes.pop.allBlocksAndValuesDefinedP

  def parse[T](
      input: ParserInputSource,
      parser: P[?] => P[T] = moduleP(using _, this),
      verboseFailures: Boolean = false,
      startIndex: Int = 0,
      instrument: fastparse.internal.Instrument = null,
  ) =
    fastparse.parse(
      input,
      parser,
      verboseFailures,
      startIndex,
      instrument,
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
  def generateOperationP[$: P](
      opName: String,
      resultsNames: Seq[String] = Seq(),
      operandsNames: Seq[String] = Seq(),
      successors: Seq[Block] = Seq(),
      properties: Map[String, Attribute] = Map(),
      regions: Seq[Region] = Seq(),
      attributes: Map[String, Attribute] = Map(),
      resultsTypes: Seq[Attribute] = Seq(),
      operandsTypes: Seq[Attribute] = Seq(),
  ): P[Operation] =

    given Parser = this

    if operandsNames.length != operandsTypes.length then
      return Fail(
        s"Number of operands (${operandsNames.length}) does not match the number of the corresponding operand types (${operandsTypes
            .length}) in \"$opName\"."
      )

    if resultsNames.length != resultsTypes.length then
      return Fail(
        s"Number of results (${resultsNames.length}) does not match the number of the corresponding result types (${resultsTypes
            .length}) in \"$opName\"."
      )

    (operandsNames zip operandsTypes).foldLeft(
      Pass(Seq.empty[Value[Attribute]])
    )((l: P[Seq[Value[Attribute]]], r: (String, Attribute)) =>
      (l ~ operandP(r._1, r._2)).map(_ :+ _)
    ).flatMap(operands =>
      (resultsNames zip resultsTypes).foldLeft(
        Pass(Seq.empty[Result[Attribute]])
      )((l: P[Seq[Result[Attribute]]], r: (String, Attribute)) =>
        (l ~ resultP(r._1, r._2)).map(_ :+ _)
      ).flatMap(results =>
        context.getOpCompanion(opName, allowUnregisteredDialect) match
          case Right(companion) =>
            Pass(
              companion(
                operands = operands,
                successors = successors,
                properties = properties,
                results = results,
                attributes = DictType.from(attributes),
                regions = regions,
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
    val inputLine = traced.input.slice(0, length).split("\n")(line - 1)

    // Build a visual indicator of where the error is.
    val indicator = " " * (col - 1) + "^"

    // Build the error message.
    val msg =
      s"Parse error at ${inputPath.getOrElse("-")}:${line +
          lineOffset}:$col:\n\n$inputLine\n$indicator\n${traced.label}"

    if parsingDiagnostics then msg
    else
      Console.err.println(msg)
      sys.exit(1)

def operandP[$: P, A <: Attribute](name: String, typ: A)(using
    p: Parser
): P[Value[A]] =
  p.scopes.collectFirst {
    case scope if scope.valueMap.contains(name) =>
      scope.valueMap(name)
  } match
    case Some(value) if value.typ == typ => Pass(value.asInstanceOf[Value[A]])
    case Some(value)                     =>
      Fail(
        s"Value %$name defined with type ${value.typ}, but used with type $typ."
      )
    case None =>
      val forwardValue = Value(typ)
      p.scopes.top.valueMap(name) = forwardValue
      p.scopes.top.forwardValues += name
      Pass(forwardValue)

def resultP[$: P, A <: Attribute](
    name: String,
    typ: A,
)(using p: Parser): P[Result[A]] =
  P(
    p.scopes.top.defineValueP(name, typ).map(_.asInstanceOf[Result[A]])
  )

/*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
|| TOP LEVEL PRODUCTION  ||
\*≡==---==≡≡≡≡≡≡≡==---==≡*/

// [x] toplevel := (operation | attribute-alias-def | type-alias-def)*
// shortened definition TODO: finish...

def moduleP[$: P](using p: Parser): P[Operation] = P(
  Start ~ p.enterRegionP ~ (operationP | attributeAliasDefP | typeAliasDefP).rep
    .map(
      _.collect { case o: Operation =>
        o
      }
    ) ~/ p.exitRegionP ~ End
).map((toplevel: Seq[Operation]) =>
  toplevel.toList match
    case (head: ModuleOp) :: Nil => head
    case (head: DerivedOperationCompanion[ModuleOp]#UnstructuredOp) :: Nil =>
      head
    case _ =>
      val block = new Block(operations = toplevel)
      val region = Region(block)
      val moduleOp = ModuleOp(region)

      for op <- toplevel do op.containerBlock = Some(block)
      block.containerRegion = Some(region)
      region.containerOperation = Some(moduleOp)

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

def operationP[$: P](using Parser): P[Operation] = P(
  opResultListP./.flatMap(resNames =>
    genericOperationP(resNames) | customOperationP(resNames)
  ) ~/ trailingLocationP.?
)./

import scala.collection.mutable.ArrayBuffer

private def genericOperandsTypesRecP[$: P](using
    expected: Int,
    p: Parser,
)(operandsNames: Seq[String], parsed: ArrayBuffer[Value[Attribute]]): P[Unit] =
  operandsNames match
    case head :: tail =>
      ",".explain(
        f"Number of operands ($expected) does not match the number of the corresponding operand types (${parsed
            .length})."
      ) ~ typeP.flatMap(operandP(head, _)).map(parsed.addOne(_): Unit) ~
        genericOperandsTypesRecP(
          tail,
          parsed,
        )
    case Nil => Pass(())

private def genericOperandsTypesP[$: P](
    operandsNames: Seq[String]
)(using Parser): P[Seq[Value[Attribute]]] =
  "(" ~ {
    operandsNames match
      case Nil          => Pass(Seq.empty[Value[Attribute]])
      case head :: tail =>
        val buffer = ArrayBuffer.empty[Value[Attribute]]
        buffer.sizeHint(operandsNames.size)
        typeP.flatMap(operandP(head, _)).map(buffer.addOne(_): Unit) ~
          genericOperandsTypesRecP(using operandsNames.length)(
            tail,
            buffer,
          ).map(_ => buffer.toSeq)
  } ~ ")"

private def genericResultsTypesRecP[$: P](using
    expected: Int,
    p: Parser,
)(resultsNames: Seq[String], parsed: ArrayBuffer[Result[Attribute]]): P[Unit] =
  resultsNames match
    case head :: tail =>
      ",".explain(
        f"Number of results ($expected) does not match the number of the corresponding operand types (${parsed
            .length})."
      ) ~ typeP.flatMap(resultP(head, _)).map(parsed.addOne(_): Unit) ~
        genericResultsTypesRecP(
          tail,
          parsed,
        )
    case Nil => Pass(())

private def genericResultsTypesP[$: P](
    resultsNames: Seq[String]
)(using Parser): P[Seq[Result[Attribute]]] =
  "(" ~ {
    resultsNames match
      case Nil          => Pass(Seq.empty[Result[Attribute]])
      case head :: tail =>
        val buffer = ArrayBuffer.empty[Result[Attribute]]
        buffer.sizeHint(resultsNames.size)
        typeP.flatMap(resultP(head, _)).map(buffer.addOne(_): Unit) ~
          genericResultsTypesRecP(using resultsNames.length)(
            tail,
            buffer,
          ).map(_ => buffer.toSeq)
  } ~ ")" | typeP.flatMap(resultP(resultsNames.head, _)).map(Seq(_))

private def genericOperationNameP[$: P](using
    p: Parser
): P[OperationCompanion[?]] =
  stringLiteralP
    .flatMap(
      p.context.getOpCompanion(_, p.allowUnregisteredDialect) match
        case Right(companion) => Pass(companion)
        case Left(error)      => Fail(error)
    )

private def genericOperationP[$: P](
    resultsNames: Seq[String]
)(using Parser): P[Operation] =
  genericOperationNameP.flatMapX((opCompanion: OperationCompanion[?]) =>
    "(" ~ operandNamesP.orElse(Seq()).flatMap((operandsNames: Seq[String]) =>
      (")" ~/ successorListP.orElse(Seq()) ~/ propertiesP.orElse(Map.empty) ~/
        regionListP.orElse(Seq()) ~/ optionalAttributesP ~/ ":" ~/
        genericOperandsTypesP(
          operandsNames
        ) ~ "->" ~/ genericResultsTypesP(resultsNames)).map(
        (
            successors: Seq[Block],
            properties: Map[String, Attribute],
            regions: Seq[Region],
            attributes: Map[String, Attribute],
            operands: Seq[Value[Attribute]],
            results: Seq[Result[Attribute]],
        ) =>
          opCompanion(
            operands,
            successors,
            results,
            regions,
            properties,
            attributes.to(DictType),
          )
      )
    )
  )

private def customOperationP[$: P](
    resNames: Seq[String]
)(using p: Parser) =
  prettyDialectReferenceNameP./.flatMapTry { (x: String, y: String) =>
    p.context.getOpCompanion(s"$x.$y") match
      case Right(companion) =>
        Pass ~ companion.parse(resNames)
      case Left(_) =>
        Fail(
          s"Operation $x.$y is not defined in any supported Dialect."
        )
  }

private def regionListP[$: P](using Parser) =
  "(" ~ regionP().rep(sep = ",") ~ ")"

// // Type aliases
// [x] type-alias-def ::= `!` alias-name `=` type
// [x] type-alias ::= `!` alias-name

private def typeAliasDefP[$: P](using p: Parser) =
  ("!" ~~ aliasNameP ~ "=" ~ typeP).flatMap((name: String, value: Attribute) =>
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

private def attributeAliasDefP[$: P](using p: Parser) =
  (
    "#" ~~ aliasNameP ~ "=" ~ attributeP
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

private def populateBlockArgsP[$: P](
    block: Block,
    args: Seq[(String, Attribute)],
)(using p: Parser) =
  args.foldLeft(Pass(Seq.empty[BlockArgument[Attribute]]))((l, r) =>
    (l ~ p.scopes.top.defineBlockArgumentP(r._1, r._2)).map(_ :+ _)
  ).map(args =>
    block.arguments ++= args
    block.arguments.foreach(_.owner = Some(block))
    block
  )

private def blockBodyP[$: P](block: Block)(using Parser) =
  operationP.map(op =>
    op.containerBlock = Some(block)
    block.operations.addOne(op): Unit
  ).rep ~ Pass(block)

def blockP[$: P](using Parser) = P(
  P(blockLabelP.flatMap(blockBodyP))
)

private def blockLabelP[$: P](using p: Parser) =
  (blockIdP.flatMap(p.scopes.top.defineBlockP) ~ (blockArgListP.orElse(Seq())))
    .flatMap(populateBlockArgsP) ~ ":"

def successorListP[$: P](using Parser) = P(
  "[" ~ successorP.rep(sep = ",") ~ "]"
)

def successorP[$: P](using p: Parser) = P(
  P(caretIdP).map(p.scopes.top.forwardBlock)
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

def regionP[$: P](
    entryArgs: Seq[(String, Attribute)] = Seq()
)(using p: Parser) = P(
  "{" ~/ p.enterRegionP ~/
    (populateBlockArgsP(new Block(), entryArgs).flatMap(blockBodyP) ~/
      blockP.rep).map((entry: Block, blocks: Seq[Block]) =>
      if entry.operations.isEmpty && entry.arguments.isEmpty then blocks
      else entry +: blocks
    ) ~/ "}" ~/ p.exitRegionP
).map(Region(_))

// [x] - value-id-and-type ::= value-id `:` type

// // Non-empty list of names and types.
// [x] - value-id-and-type-list ::= value-id-and-type (`,` value-id-and-type)*

// [x] - block-arg-list ::= `(` value-id-and-type-list? `)`

def valueIdAndTypeP[$: P](using Parser) = P(valueIdP ~ ":" ~ typeP)

private def valueIdAndTypeListP[$: P](using Parser) =
  P(valueIdAndTypeP.rep(sep = ",")).orElse(Seq())

private def blockArgListP[$: P](using Parser) =
  P(
    "(" ~ valueIdAndTypeP.rep(sep = ",") ~ ")"
  )

// [x] dictionary-properties ::= `<` dictionary-attribute `>`
// [x] dictionary-attribute  ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`

/** Parses a properties dictionary, which synctatically simply is an attribute
  * dictionary wrapped in angle brackets.
  *
  * @return
  *   A properties dictionary parser.
  */
def propertiesP[$: P](using Parser) = P(
  "<" ~ attributeDictionaryP ~ ">"
)

/** Parses an attributes dictionary.
  *
  * @return
  *   An attribute dictionary parser.
  */
def attributeDictionaryP[$: P](using
    Parser
): P[Map[String, Attribute]] = P(
  "{" ~ attributeEntryP.rep(sep = ",").map(Map.from) ~ "}"
)

/** Parses an optional properties dictionary from the input.
  *
  * @return
  *   An optional dictionary of properties - empty if no dictionary is present.
  */
def optionalPropertiesP[$: P](using Parser) =
  (propertiesP).orElse(DictType.empty)

/** Parses an optional attributes dictionary from the input.
  *
  * @return
  *   An optional dictionary of attributes - empty if no dictionary is present.
  */
def optionalAttributesP[$: P](using Parser) =
  (attributeDictionaryP).orElse(Map.empty)

/** Parses an optional attributes dictionary from the input, preceded by the
  * `attributes` keyword.
  *
  * @return
  *   An optional dictionary of attributes - empty if no keyword is present.
  */
def optionalKeywordAttributesP[$: P](using Parser) =
  ("attributes" ~/ attributeDictionaryP).orElse(Map.empty)

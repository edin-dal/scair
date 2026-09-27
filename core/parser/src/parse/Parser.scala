package scair.parse

import fastparse.*
import fastparse.Parsed.Failure
import fastparse.internal.Util
import scair.MLContext
import scair.clair.OpDefs
import scair.dialects.builtin.ModuleOp
import scair.ir.*

import scala.collection.mutable

// ██████╗░ ░█████╗░ ██████╗░ ░██████╗ ███████╗ ██████╗░
// ██╔══██╗ ██╔══██╗ ██╔══██╗ ██╔════╝ ██╔════╝ ██╔══██╗
// ██████╔╝ ███████║ ██████╔╝ ╚█████╗░ █████╗░░ ██████╔╝
// ██╔═══╝░ ██╔══██║ ██╔══██╗ ░╚═══██╗ ██╔══╝░░ ██╔══██╗
// ██║░░░░░ ██║░░██║ ██║░░██║ ██████╔╝ ███████╗ ██║░░██║
// ╚═╝░░░░░ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ╚═════╝░ ╚══════╝ ╚═╝░░╚═╝

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
      val newBlock = Block()
      blockMap(blockName) = newBlock
      Pass(newBlock)

  inline def forwardBlock(
      blockName: String
  ): Block =
    blockMap.getOrElseUpdate(
      blockName, {
        forwardBlocks += blockName
        Block()
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
    .orElse(Seq.empty)

private inline def sequenceValues(
    name: String,
    no: BigInt,
): Seq[String] = (0 to (no.toInt - 1)).map(no => s"$name#$no")

private inline def opResultP[$: P] = (valueIdP.flatMapX(name =>
  (":" ~~ decDigitsP.!.map(d => sequenceValues(name, d.toInt)))
    .orElse(Seq(name))
))

private def locationNumberP[$: P]: P[Int] = decDigitsP.!.mapTry(_.toInt)

private def fileLocationP[$: P]: P[Location] =
  (stringLiteralP ~ ":" ~ locationNumberP ~ ":" ~ locationNumberP)
    .flatMap { (filename, line, column) =>
      ("to" ~/ locationNumberP.? ~ ":" ~ locationNumberP).?.map {
        case Some((endLine, endColumn)) =>
          FileLineColRange(
            filename,
            line,
            column,
            endLine.getOrElse(line),
            endColumn,
          )
        case None => FileLineColLoc(filename, line, column)
      }
    }

private def trailingLocationP[$: P]: P[Location] =
  "loc" ~/ "(" ~ ("unknown".map(_ => UnknownLoc) | fileLocationP) ~ ")"

/*≡==--==≡≡≡≡≡≡≡≡==--=≡≡*\
||     PARSER CLASS     ||
\*≡==---==≡≡≡≡≡≡==---==≡*/

final class MLIRParser(
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
    private[parse] final val inputLineOffset: Int = 0,
    private[parse] final val sourceLocations: Boolean = false,
) extends Parser:

  /*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
  ||   INTERFACE IMPLEMENTATION   ||
  \*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

  override def attributeP[$: P]: P[Attribute] =
    attributeImplP(using summon, this)

  override def typeP[$: P]: P[Attribute] = typeImplP(using summon, this)

  override def typeListP[$: P]: P[Seq[Attribute]] =
    typeListImplP(using summon, this)

  override def parenTypeListP[$: P]: P[Seq[Attribute]] =
    parenTypeListImplP(using summon, this)

  override def regionP[$: P](entryArgs: Seq[(String, Attribute)]): P[Region] =
    regionImplP(entryArgs)(using summon, this)

  override def operandP[$: P, A <: Attribute](
      name: String,
      typ: A,
  ): P[Value[A]] =
    operandImplP(name, typ)(using summon, this)

  override def resultP[$: P, A <: Attribute](
      name: String,
      typ: A,
  ): P[Result[A]] =
    resultImplP(name, typ)(using summon, this)

  override def valueIdAndTypeP[$: P]: P[(String, Attribute)] =
    valueIdAndTypeImplP(using summon, this)

  override def attributeDictionaryP[$: P]: P[Map[String, Attribute]] =
    attributeDictionaryImplP(using summon, this)

  override def optionalAttributesP[$: P]: P[Map[String, Attribute]] =
    optionalAttributesImplP(using summon, this)

  override def operationP[$: P]: P[Operation] =
    operationImplP(using summon, this)

  override def moduleP[$: P]: P[Operation] = moduleImplP(using summon, this)

  // Line starts of the last indexed input, cached across calls.
  private var lineStartsOf: String | Null = null
  private var lineStarts: Array[Int] = Array.empty

  // Same numbering as prettyIndex, but binary-searched over cached line starts.
  private[parse] def sourceLocation[$: P as ctx](index: Int): Location =
    ctx.input match
      case IndexedParserInput(data) if sourceLocations =>
        if !(lineStartsOf eq data) then
          lineStarts = Util.lineNumberLookup(data)
          lineStartsOf = data
        val found = java.util.Arrays.binarySearch(lineStarts, index)
        val line = math.max(0, if found >= 0 then found else -found - 2)
        FileLineColLoc(
          inputPath.getOrElse("-"),
          line + 1 + inputLineOffset,
          index - lineStarts(line) + 1,
        )
      case _ => UnknownLoc

  private[parse] def enterRegionP[$: P] =
    scopes.push(new Scope())
    Pass

  private[parse] def exitRegionP[$: P] =
    scopes.pop().allBlocksAndValuesDefinedP

  override def parse[T](
      input: ParserInputSource,
      parser: P[?] => P[T],
      verboseFailures: Boolean,
      startIndex: Int,
      instrument: fastparse.internal.Instrument,
  ): Parsed[T] =
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
  override def generateOperationP[$: P](
      opName: String,
      resultsNames: Seq[String],
      operandsNames: Seq[String],
      successors: Seq[Block],
      properties: Map[String, Attribute],
      regions: Seq[Region],
      attributes: Map[String, Attribute],
      resultsTypes: Seq[Attribute],
      operandsTypes: Seq[Attribute],
  ): P[Operation] =

    given MLIRParser = this

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
      (l ~ operandImplP(r._1, r._2)).map(_ :+ _)
    ).flatMap(operands =>
      (resultsNames zip resultsTypes).foldLeft(
        Pass(Seq.empty[Result[Attribute]])
      )((l: P[Seq[Result[Attribute]]], r: (String, Attribute)) =>
        (l ~ resultImplP(r._1, r._2)).map(_ :+ _)
      ).flatMap(results =>
        context.getOpCompanion(opName, allowUnregisteredDialect) match
          case Right(companion) =>
            Pass(
              companion(
                operands = operands,
                successors = successors,
                properties = properties,
                results = results,
                attributes = attributes,
                regions = regions,
              )
            )
          case Left(error) => Fail(error)
      )
    )

  override def error(failure: Failure, lineOffset: Int): String =
    // .trace() below reparses from the start with more bookkeeping to provide helpful
    // context for the error message.
    // We do this very non-functional bookkeeping in currentScope ourselves, which
    // is disfunctional with this behaviour; it then reparses everything with the state
    // it already had at the time of the catched error!
    // This is a workaround to get the error message with the correct state.
    // TODO: More functional and fastparse-compatible state handling!
    scopes.popAll()
    scopes.push(new Scope())
    attributeAliases.clear()
    typeAliases.clear()

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

def operandImplP[$: P, A <: Attribute](name: String, typ: A)(using
    p: MLIRParser
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

def resultImplP[$: P, A <: Attribute](
    name: String,
    typ: A,
)(using p: MLIRParser): P[Result[A]] =
  P(
    p.scopes.top.defineValueP(name, typ).map(_.asInstanceOf[Result[A]])
  )

/*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
|| TOP LEVEL PRODUCTION  ||
\*≡==---==≡≡≡≡≡≡≡==---==≡*/

// [x] toplevel := (operation | attribute-alias-def | type-alias-def)*
// shortened definition TODO: finish...

def moduleImplP[$: P](using p: MLIRParser): P[Operation] = P(
  Start ~ p.enterRegionP ~ (operationImplP | attributeAliasDefP | typeAliasDefP)
    .rep
    .map(
      _.collect { case o: Operation =>
        o
      }
    ) ~/ p.exitRegionP ~ End
).map((toplevel: Seq[Operation]) =>
  toplevel.toList match
    case (head: ModuleOp) +: Nil                        => head
    case (head: OpDefs[ModuleOp]#UnstructuredOp) +: Nil =>
      head
    case _ =>
      val block = Block(operations = toplevel)
      val region = Region(block)
      val moduleOp = ModuleOp(region)
      if p.sourceLocations then
        moduleOp.at(FileLineColLoc(p.inputPath.getOrElse("-"), 0, 0))

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

def operationImplP[$: P](using p: MLIRParser): P[Operation] = P(
  opResultListP./.flatMap(resNames =>
    (Index.map(p.sourceLocation(_)) ~~
      (genericOperationP(resNames) | customOperationP(resNames)))
      .map((location, op) => op.at(location))
  ) ~/ trailingLocationP.?
).map { (op, location) =>
  location.foreach(op.at)
  op
}./

def genericOperandsTypesP[$: P](
    operandsNames: Seq[String]
)(using MLIRParser): P[Seq[Value[Attribute]]] =
  val error = (i: Int) =>
    f"Number of operands (${operandsNames.size}) does not match the number of the corresponding operand types ($i)."
  "(" ~ operandsNames.flatRep(
    name => typeImplP.flatMap(operandImplP(name, _)),
    sep = ",",
    error = error,
  ).flatMap(types =>
    ")".explain(
      f"Number of operands (${operandsNames.size}) does not match the number of the corresponding operand types."
    ).map(_ => types)
  )

private def genericResultsTypesP[$: P](
    resultsNames: Seq[String]
)(using MLIRParser): P[Seq[Result[Attribute]]] =
  val error = (i: Int) =>
    f"Number of results (${resultsNames.size}) does not match the number of the corresponding result types ($i)."
  "(" ~ resultsNames.flatRep(
    name => typeImplP.flatMap(resultImplP(name, _)),
    sep = ",",
    error = error,
  ).flatMap(types =>
    ")".explain(
      f"Number of results (${resultsNames.size}) does not match the number of the corresponding result types."
    ).map(_ => types)
  ) | typeImplP.flatMap(resultImplP(resultsNames.head, _)).map(Seq(_))

private def genericOperationNameP[$: P](using
    p: MLIRParser
): P[OperationCompanion[?]] =
  stringLiteralP./
    .flatMap(
      p.context.getOpCompanion(_, p.allowUnregisteredDialect) match
        case Right(companion) => Pass(companion)
        case Left(error)      => Fail(error)
    )

private def genericOperationP[$: P](
    resultsNames: Seq[String]
)(using MLIRParser): P[Operation] =
  genericOperationNameP.flatMap((opCompanion: OperationCompanion[?]) =>
    "(" ~ operandNamesP.orElse(Seq.empty)
      .flatMap((operandsNames: Seq[String]) =>
        ")" ~/ successorListP.orElse(Seq.empty).flatMap(successors =>
          propertiesP.orElse(Map.empty).flatMap(properties =>
            regionListP.orElse(Seq.empty).flatMap(regions =>
              optionalAttributesImplP.flatMap(attributes =>
                ":" ~/ genericOperandsTypesP(
                  operandsNames
                ).flatMap(operands =>
                  ("->" ~/ genericResultsTypesP(resultsNames))
                    .map((results: Seq[Result[Attribute]]) =>
                      opCompanion(
                        operands,
                        successors,
                        results,
                        regions,
                        properties,
                        attributes,
                      )
                    )
                )
              )
            )
          )
        )
      )
  )

private def customOperationP[$: P](
    resNames: Seq[String]
)(using p: MLIRParser) =
  prettyDialectReferenceNameP./.flatMapTry { (x: String, y: String) =>
    p.context.getOpCompanion(s"$x.$y") match
      case Right(companion) =>
        Pass ~ companion.parse(resNames)
      case Left(_) =>
        Fail(
          s"Operation $x.$y is not defined in any supported Dialect."
        )
  }

private def regionListP[$: P](using MLIRParser) =
  "(" ~ regionImplP().rep(sep = ",") ~ ")"

// // Type aliases
// [x] type-alias-def ::= `!` alias-name `=` type
// [x] type-alias ::= `!` alias-name

private def typeAliasDefP[$: P](using p: MLIRParser) =
  ("!" ~~ aliasNameP ~ "=" ~ typeImplP)
    .flatMap((name: String, value: Attribute) =>
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

private def attributeAliasDefP[$: P](using p: MLIRParser) =
  (
    "#" ~~ aliasNameP ~ "=" ~ attributeImplP
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
)(using p: MLIRParser) =
  args.foldLeft(Pass(Seq.empty[BlockArgument[Attribute]]))((l, r) =>
    (l ~ p.scopes.top.defineBlockArgumentP(r._1, r._2)).map(_ :+ _)
  ).map(args =>
    block.arguments ++= args
    block.arguments.foreach(_.owner = Some(block))
    block
  )

private def blockBodyP[$: P](block: Block)(using MLIRParser) =
  // TODO: temporary solution to populate indexes within block body.
  var idx = 0
  operationImplP.map(op =>
    op.containerBlock = Some(block)
    block.operations.addOne(op): Unit
    op.blockIndex = idx
    idx += 1
  ).rep ~ Pass(block)

def blockP[$: P](using MLIRParser) = P(
  P(blockLabelP.flatMap(blockBodyP))
)

private def blockLabelP[$: P](using p: MLIRParser) =
  (blockIdP.flatMap(p.scopes.top.defineBlockP) ~
    (blockArgListP.orElse(Seq.empty))).flatMap(populateBlockArgsP) ~ ":"

def successorListP[$: P](using MLIRParser) = P(
  "[" ~ successorP.rep(sep = ",") ~ "]"
)

def successorP[$: P](using p: MLIRParser) = P(
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

def regionImplP[$: P](
    entryArgs: Seq[(String, Attribute)] = Seq.empty
)(using p: MLIRParser) = P(
  "{" ~/ p.enterRegionP ~/
    (populateBlockArgsP(Block(), entryArgs).flatMap(blockBodyP) ~/ blockP.rep)
      .map((entry: Block, blocks: Seq[Block]) =>
        if entry.operations.isEmpty && entry.arguments.isEmpty then blocks
        else entry +: blocks
      ) ~/ "}" ~/ p.exitRegionP
).map(Region(_))

// [x] - value-id-and-type ::= value-id `:` type

// // Non-empty list of names and types.
// [x] - value-id-and-type-list ::= value-id-and-type (`,` value-id-and-type)*

// [x] - block-arg-list ::= `(` value-id-and-type-list? `)`

def valueIdAndTypeImplP[$: P](using MLIRParser) = P(valueIdP ~ ":" ~ typeImplP)

private def valueIdAndTypeListP[$: P](using MLIRParser) =
  P(valueIdAndTypeImplP.rep(sep = ",")).orElse(Seq.empty)

private def blockArgListP[$: P](using MLIRParser) =
  P(
    "(" ~ valueIdAndTypeImplP.rep(sep = ",") ~ ")"
  )

// [x] dictionary-properties ::= `<` dictionary-attribute `>`
// [x] dictionary-attribute  ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`

/** Parses a properties dictionary, which synctatically simply is an attribute
  * dictionary wrapped in angle brackets.
  *
  * @return
  *   A properties dictionary parser.
  */
def propertiesP[$: P](using MLIRParser) = P(
  "<" ~ attributeDictionaryImplP ~ ">"
)

/** Parses an attributes dictionary.
  *
  * @return
  *   An attribute dictionary parser.
  */
def attributeDictionaryImplP[$: P](using
    MLIRParser
): P[Map[String, Attribute]] = P(
  "{" ~ attributeEntryP.rep(sep = ",").map(Map.from) ~ "}"
)

/** Parses an optional properties dictionary from the input.
  *
  * @return
  *   An optional dictionary of properties - empty if no dictionary is present.
  */
def optionalPropertiesP[$: P](using MLIRParser) =
  (propertiesP).orElse(Map.empty)

/** Parses an optional attributes dictionary from the input.
  *
  * @return
  *   An optional dictionary of attributes - empty if no dictionary is present.
  */
def optionalAttributesImplP[$: P](using MLIRParser) =
  (attributeDictionaryImplP).orElse(Map.empty)

/** Parses an optional attributes dictionary from the input, preceded by the
  * `attributes` keyword.
  *
  * @return
  *   An optional dictionary of attributes - empty if no keyword is present.
  */
def optionalKeywordAttributesP[$: P](using MLIRParser) =
  ("attributes" ~/ attributeDictionaryImplP).orElse(Map.empty)

/** Creates the default, fastparse-based, [[Parser]] implementation. */
def Parser(
    context: MLContext,
    inputPath: Option[String] = None,
    parsingDiagnostics: Boolean = false,
    allowUnregisteredDialect: Boolean = false,
    inputLineOffset: Int = 0,
    sourceLocations: Boolean = false,
): MLIRParser =
  new MLIRParser(
    context = context,
    inputPath = inputPath,
    parsingDiagnostics = parsingDiagnostics,
    allowUnregisteredDialect = allowUnregisteredDialect,
    inputLineOffset = inputLineOffset,
    sourceLocations = sourceLocations,
  )

package scair.clair.macros

import fastparse.P
import scair.Printer
import scair.ir.*
import scair.parse.Parser
import scair.utils.*

import scala.quoted.*
import scala.util.Failure
import scala.util.Success
import scala.util.Try

// ████████╗ ██╗░░░██╗ ██████╗░ ███████╗
// ╚══██╔══╝ ╚██╗░██╔╝ ██╔══██╗ ██╔════╝
// ░░░██║░░░ ░╚████╔╝░ ██████╔╝ █████╗░░
// ░░░██║░░░ ░░╚██╔╝░░ ██╔═══╝░ ██╔══╝░░
// ░░░██║░░░ ░░░██║░░░ ██║░░░░░ ███████╗
// ░░░╚═╝░░░ ░░░╚═╝░░░ ╚═╝░░░░░ ╚══════╝
//
// ░█████╗░ ██╗░░░░░ ░█████╗░ ░██████╗ ░██████╗ ███████╗ ░██████╗
// ██╔══██╗ ██║░░░░░ ██╔══██╗ ██╔════╝ ██╔════╝ ██╔════╝ ██╔════╝
// ██║░░╚═╝ ██║░░░░░ ███████║ ╚█████╗░ ╚█████╗░ █████╗░░ ╚█████╗░
// ██║░░██╗ ██║░░░░░ ██╔══██║ ░╚═══██╗ ░╚═══██╗ ██╔══╝░░ ░╚═══██╗
// ╚█████╔╝ ███████╗ ██║░░██║ ██████╔╝ ██████╔╝ ███████╗ ██████╔╝
// ░╚════╝░ ╚══════╝ ╚═╝░░╚═╝ ╚═════╝░ ╚═════╝░ ╚══════╝ ╚═════╝░

trait AttributeCustomParser[T <: Attribute]:
  export scair.parse.whitespace

  def parse[$: P](using
      Parser
  ): P[T]

trait DerivedAttributeCompanion[T <: Attribute] extends AttributeCompanion[T]:
  def parameters(attr: T): Seq[Attribute | Seq[Attribute]]
  override def parse[$: P](using Parser): P[T]

/** Object for deriving attribute companions. */
object DerivedAttributeCompanion:

  /** Derives a DerivedAttributeCompanion for an attribute type.
    *
    * @tparam T
    *   The attribute type.
    * @return
    *   The derived companion.
    */
  inline def derived[T <: Attribute]: DerivedAttributeCompanion[T] = ${
    derivedAttributeCompanion[T]
  }

/** Type class for custom operation parsers. Implement this to provide a custom
  * parser for your operation type instead of the default generated one.
  *
  * @tparam T
  *   The operation type to parse.
  */
trait OperationCustomParser[T <: Operation]:
  export scair.parse.whitespace

  /** Parses an instance of the operation from input.
    *
    * @param resNames
    *   Names for the operation's results.
    * @return
    *   A fastparse parser for the operation.
    */
  def parse[$: P](
      resNames: Seq[String]
  )(using Parser): P[T]

/** Companion type class for derived operations. Provides all the capabilities
  * needed to work with a derived operation: parsing, printing, verification,
  * structuring, destructuring, etc.
  *
  * @tparam T
  *   The operation type.
  */
trait DerivedOperationCompanion[T <: Operation] extends OperationCompanion[T]:

  companion =>

  /** Extracts operands from an operation instance. */
  def operands(adtOp: T): Seq[Value[Attribute]]

  /** Extracts successors from an operation instance. */
  def successors(adtOp: T): Seq[Block]

  /** Extracts results from an operation instance. */
  def results(adtOp: T): Seq[Result[Attribute]]

  /** Extracts regions from an operation instance. */
  def regions(adtOp: T): Seq[Region]

  /** Extracts properties from an operation instance. */
  def properties(adtOp: T): Map[String, Attribute]

  /** Prints the operation using its custom format. */
  def customPrint(adtOp: T, p: Printer)(using indentLevel: Int): Unit

  /** Verifies the operation's constraints. */
  def constraintVerify(adtOp: T): OK[Operation]

  /** Unstructured representation of an operation. Used when the structured
    * representation cannot be constructed (e.g., due to invalid operands).
    */

  case class UnstructuredOp(
      override val operands: Seq[Value[Attribute]] = Seq(),
      override val successors: Seq[Block] = Seq(),
      override val results: Seq[Result[Attribute]] = Seq(),
      override val regions: Seq[Region] = Seq(),
      override val properties: Map[String, Attribute] = Map
        .empty[String, Attribute],
      override val attributes: DictType[String, Attribute] = DictType
        .empty[String, Attribute],
  ) extends Operation:

    override def updated(
        operands: Seq[Value[Attribute]] = operands,
        successors: Seq[Block] = successors,
        results: Seq[Result[Attribute]] = results.map(_.typ).map(Result(_)),
        regions: Seq[Region] = detachedRegions,
        properties: Map[String, Attribute] = properties,
        attributes: DictType[String, Attribute] = attributes,
    ): Operation =
      UnstructuredOp(
        operands,
        successors,
        results,
        regions,
        properties,
        attributes,
      )

    override def structured = Try(companion.structure(this)) match
      case Failure(e)  => Err(e.toString())
      case Success(op) => op.asInstanceOf[Operation].structured

    override def verify(): OK[Operation] =
      structured.flatMap(op => op.verify())

    override def name = companion.name

  /** Constructs an operation from its components. Attempts to create a
    * structured representation, falling back to unstructured if necessary.
    *
    * @param operands
    *   The operation's operands.
    * @param successors
    *   The operation's successors.
    * @param results
    *   The operation's results.
    * @param regions
    *   The operation's regions.
    * @param properties
    *   The operation's properties.
    * @param attributes
    *   The operation's attributes.
    * @return
    *   Either a structured or unstructured operation.
    */
  def apply(
      operands: Seq[Value[Attribute]] = Seq(),
      successors: Seq[scair.ir.Block] = Seq(),
      results: Seq[Result[Attribute]] = Seq(),
      regions: Seq[Region] = Seq(),
      properties: Map[String, Attribute] = Map.empty[String, Attribute],
      attributes: DictType[String, Attribute] = DictType
        .empty[String, Attribute],
  ): UnstructuredOp | T & Operation

  /** Converts a structured operation to its unstructured representation.
    *
    * @param adtOp
    *   The structured operation.
    * @return
    *   The unstructured representation.
    */
  def destructure(adtOp: T): UnstructuredOp

  /** Converts an unstructured operation to its structured representation.
    *
    * @param unstrucOp
    *   The unstructured operation.
    * @return
    *   The structured operation.
    * @throws Exception
    *   If the unstructured operation cannot be structured.
    */
  def structure(unstrucOp: UnstructuredOp): T

/** Object for deriving operation companions. */
object DerivedOperationCompanion:

  /** Derives a DerivedOperationCompanion for an operation type.
    *
    * @tparam T
    *   The operation type.
    * @return
    *   The derived companion.
    */
  inline def derived[T <: Operation]: DerivedOperationCompanion[T] = ${
    deriveOperationCompanion[T]
  }

/** Recursively summons operation companions for a tuple of operation types.
  *
  * @tparam T
  *   Tuple of operation types.
  * @return
  *   Sequence of companion expressions.
  */

def summonOperationCompanionsMacroRec[T <: Tuple: Type](using
    Quotes
): Seq[Expr[OperationCompanion[?]]] =
  import quotes.reflect.*
  Type.of[T] match
    case '[type o <: Operation; o *: ts] =>
      val dat = Expr.summon[OperationCompanion[o]]
        .getOrElse(
          report
            .errorAndAbort(
              f"Could not summon OperationCompanion for ${Type.show[o]}"
            )
        )
      dat +: summonOperationCompanionsMacroRec[ts]

    case '[EmptyTuple] => Seq()

def summonOperationCompanionsMacro[T <: Tuple: Type](using
    Quotes
): Expr[Seq[OperationCompanion[?]]] =
  Expr.ofSeq(summonOperationCompanionsMacroRec[T])

/** Recursively summons attribute companions for a tuple of attribute types.
  *
  * @tparam T
  *   Tuple of attribute types.
  * @return
  *   Sequence of companion expressions.
  */

def summonAttributeCompanionsMacroRec[T <: Tuple: Type](using
    Quotes
): Seq[Expr[AttributeCompanion[?]]] =
  import quotes.reflect.*
  Type.of[T] match
    case '[type a <: Attribute; `a` *: ts] =>
      val dat = Expr.summon[AttributeCompanion[a]]
        .getOrElse(
          report
            .errorAndAbort(
              f"Could not summon AttributeCompanion for ${Type.show[a]}"
            )
        )
      dat +: summonAttributeCompanionsMacroRec[ts]
    case '[EmptyTuple] => Seq()

def summonAttributeCompanionsMacro[T <: Tuple: Type](using
    Quotes
): Expr[Seq[AttributeCompanion[?]]] =
  Expr.ofSeq(summonAttributeCompanionsMacroRec[T])

/** Summons attribute companions at compile time.
  *
  * @tparam T
  *   Tuple of attribute types.
  * @return
  *   Sequence of attribute companions.
  */

inline def summonAttributeCompanions[T <: Tuple]: Seq[AttributeCompanion[?]] =
  ${ summonAttributeCompanionsMacro[T] }

/** Summons operation companions at compile time.
  *
  * @tparam T
  *   Tuple of operation types.
  * @return
  *   Sequence of operation companions.
  */

inline def summonOperationCompanions[T <: Tuple]: Seq[OperationCompanion[?]] =
  ${ summonOperationCompanionsMacro[T] }

/** Summons a complete dialect with operations and attributes.
  *
  * @tparam Attributes
  *   Tuple of attribute types.
  * @tparam Operations
  *   Tuple of operation types.
  * @return
  *   A Dialect containing all the specified operations and attributes.
  */

inline def summonDialect[Attributes <: Tuple, Operations <: Tuple]: Dialect =
  Dialect(
    summonOperationCompanions[Operations],
    summonAttributeCompanions[Attributes],
  )

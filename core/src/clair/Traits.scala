package scair.clair.macros

import scair.Printer
import scair.ir.*
import scair.utils.*

// ████████╗ ██████╗░ ░█████╗░ ██╗ ████████╗ ░██████╗
// ╚══██╔══╝ ██╔══██╗ ██╔══██╗ ██║ ╚══██╔══╝ ██╔════╝
// ░░░██║░░░ ██████╔╝ ███████║ ██║ ░░░██║░░░ ╚█████╗░
// ░░░██║░░░ ██╔══██╗ ██╔══██║ ██║ ░░░██║░░░ ░╚═══██╗
// ░░░██║░░░ ██║░░██║ ██║░░██║ ██║ ░░░██║░░░ ██████╔╝
// ░░░╚═╝░░░ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ╚═╝ ░░░╚═╝░░░ ╚═════╝░

/** Base trait for derived attributes. Provides automatic implementation of
  * attribute names and parameters based on the ADT structure.
  *
  * To use, extend this trait with your attribute case class and specify the
  * attribute name as a type parameter: {{{case class MyAttr(value: IntegerType)
  * extends DerivedAttribute["my.attr", MyAttr]}}}
  *
  * @tparam name
  *   The fully qualified name of the attribute (e.g., "dialect.attribute").
  * @tparam T
  *   The self-type of the implementing attribute.
  * @param comp
  *   The implicitly provided companion implementing attribute operations.
  */
transparent trait DerivedAttribute[name <: String, T <: Attribute](using
    private final val comp: DerivedAttributeCompanion[T]
) extends ParametrizedAttribute:

  this: T =>

  override val name: String = comp.name

  override val parameters: Seq[Attribute | Seq[Attribute]] =
    comp.parameters(this)

/** Marker trait for types that specify a custom assembly format. The format
  * string is used to generate specialized parsers and printers.
  *
  * @tparam format
  *   The assembly format string (e.g., "$operand `:` type($operand)").
  */
trait AssemblyFormat[format <: String]

/** Base class for derived operations. Provides automatic implementation of
  * operation methods based on the ADT structure.
  *
  * To use, extend this class with your operation case class and specify the
  * operation name as a type parameter: {{{case class MyOp(operand:
  * Operand[IntegerType]) extends DerivedOperation["my.op", MyOp]}}}
  *
  * @tparam name
  *   The fully qualified name of the operation (e.g., "dialect.operation").
  * @tparam T
  *   The self-type of the implementing operation.
  * @param comp
  *   The implicitly provided companion implementing operation methods.
  */
abstract class DerivedOperation[name <: String, T <: Operation](using
    private final val comp: DerivedOperationCompanion[T]
) extends Operation:

  this: T =>

  /** Updates this operation with new components. Returns an unstructured or
    * structured operation based on what the companion can construct.
    *
    * @param operands
    *   New operands.
    * @param successors
    *   New successors.
    * @param results
    *   New results.
    * @param regions
    *   New regions.
    * @param properties
    *   New properties.
    * @param attributes
    *   New attributes.
    * @return
    *   The updated operation.
    */
  override def updated(
      operands: Seq[Value[Attribute]],
      successors: Seq[Block],
      results: Seq[Result[Attribute]],
      regions: Seq[Region],
      properties: Map[String, Attribute],
      attributes: DictType[String, Attribute],
  ) =
    comp(
      operands = operands,
      successors = successors,
      results = results,
      regions = detachedRegions,
      properties = properties,
      attributes = attributes,
    )

  /** The fully qualified name of this operation. */
  def name: String = comp.name

  /** The operands (input values) of this operation. */
  def operands: Seq[Value[Attribute]] = comp.operands(this)

  /** The successors (control flow targets) of this operation. */
  def successors: Seq[Block] = comp.successors(this)

  /** The results (output values) of this operation. */
  def results: Seq[Result[Attribute]] = comp.results(this)

  /** The regions (nested control flow) of this operation. */
  def regions: Seq[Region] = comp.regions(this)

  /** The properties (compile-time attributes) of this operation. */
  def properties: Map[String, Attribute] = comp.properties(this)

  /** Custom printer implementation. Uses the assembly format if defined. */
  override def customPrint(p: Printer)(using indentLevel: Int): Unit =
    comp.customPrint(this, p)

  /** Verifies the operation. Checks both base operation constraints and
    * type-specific constraints.
    */
  override def verify(): OK[Operation] =
    super.verify().flatMap(_ => constraintVerify())

  /** Verifies type-specific constraints on this operation. */
  def constraintVerify(): OK[Operation] =
    comp.constraintVerify(this)

package scair.clair.codegen

import scair.clair.macros.AssemblyFormatDirective
import scair.core.constraints.ConstraintImpl

import scala.quoted.*
import scala.reflect.*

// ░█████╗░ ██╗░░░░░ ░█████╗░ ██╗ ██████╗░ ██╗░░░██╗ ██████╗░
// ██╔══██╗ ██║░░░░░ ██╔══██╗ ██║ ██╔══██╗ ██║░░░██║ ╚════██╗
// ██║░░╚═╝ ██║░░░░░ ███████║ ██║ ██████╔╝ ╚██╗░██╔╝ ░░███╔═╝
// ██║░░██╗ ██║░░░░░ ██╔══██║ ██║ ██╔══██╗ ░╚████╔╝░ ██╔══╝░░
// ╚█████╔╝ ███████╗ ██║░░██║ ██║ ██║░░██║ ░░╚██╔╝░░ ███████╗
// ░╚════╝░ ╚══════╝ ╚═╝░░╚═╝ ╚═╝ ╚═╝░░╚═╝ ░░░╚═╝░░░ ╚══════╝

// ░█████╗░ ░█████╗░ ██████╗░ ███████╗ ░██████╗░ ███████╗ ███╗░░██╗
// ██╔══██╗ ██╔══██╗ ██╔══██╗ ██╔════╝ ██╔════╝░ ██╔════╝ ████╗░██║
// ██║░░╚═╝ ██║░░██║ ██║░░██║ █████╗░░ ██║░░██╗░ █████╗░░ ██╔██╗██║
// ██║░░██╗ ██║░░██║ ██║░░██║ ██╔══╝░░ ██║░░╚██╗ ██╔══╝░░ ██║╚████║
// ╚█████╔╝ ╚█████╔╝ ██████╔╝ ███████╗ ╚██████╔╝ ███████╗ ██║░╚███║
// ░╚════╝░ ░╚════╝░ ╚═════╝░ ╚══════╝ ░╚═════╝░ ╚══════╝ ╚═╝░░╚══╝

/*≡≡=-=≡≡≡≡=-=≡≡*\
||  CONTAINERS  ||
\*≡=--==≡≡==--=≡*/

/** Extractor for OpInputDef that matches any operation input definition.
  *
  * @param d
  *   The value to match.
  * @return
  *   Some if d is an OpInputDef, None otherwise.
  */
object OpInputDef:

  def unapply(d: Any) = d match
    case d: OpInputDef => Some((name = d.name))
    case _             => None

/** Base trait for all operation input definitions. An input is any component of
  * an operation (operands, results, regions, successors, properties).
  */
sealed trait OpInputDef:
  /** The name of this input definition. */
  def name: String

/** Extractor for MayVariadicOpInputDef that matches variadic/optional/single
  * input definitions.
  *
  * @param d
  *   The value to match.
  * @return
  *   Some with name and variadicity if d is a MayVariadicOpInputDef, None
  *   otherwise.
  */
object MayVariadicOpInputDef:

  def unapply(d: Any) = d match
    case d: MayVariadicOpInputDef =>
      Some((name = d.name, variadicity = d.variadicity))
    case _ => None

/** Base trait for operation input definitions that support variadicity (single,
  * variadic, or optional).
  */
sealed trait MayVariadicOpInputDef extends OpInputDef:
  /** The variadicity of this input (Single, Variadic, or Optional). */
  def variadicity: Variadicity

/** Enumeration of variadicity types for operation inputs.
  *
  *   - Single: Exactly one instance (e.g., one operand).
  *   - Variadic: Zero or more instances (e.g., variable number of operands).
  *   - Optional: Zero or one instance (e.g., optional operand).
  */
enum Variadicity:
  case Single, Variadic, Optional

/** Definition of an operand (input value) for an operation.
  *
  * @param name
  *   The name of the operand.
  * @param tpe
  *   The expected type of the operand.
  * @param variadicity
  *   Whether this is a single, variadic, or optional operand.
  * @param constraint
  *   Optional type constraint on the operand.
  */
case class OperandDef(
    override val name: String,
    val tpe: Type[?],
    override val variadicity: Variadicity = Variadicity.Single,
    val constraint: Option[Expr[ConstraintImpl[?]]] = None,
) extends OpInputDef
    with MayVariadicOpInputDef {}

/** Definition of a result (output value) for an operation.
  *
  * @param name
  *   The name of the result.
  * @param tpe
  *   The expected type of the result.
  * @param variadicity
  *   Whether this is a single, variadic, or optional result.
  * @param constraint
  *   Optional type constraint on the result.
  */
case class ResultDef(
    override val name: String,
    val tpe: Type[?],
    override val variadicity: Variadicity = Variadicity.Single,
    val constraint: Option[Expr[ConstraintImpl[?]]] = None,
) extends OpInputDef
    with MayVariadicOpInputDef {}

/** Definition of a region (nested control flow) for an operation.
  *
  * @param name
  *   The name of the region.
  * @param variadicity
  *   Whether this is a single, variadic, or optional region.
  */
case class RegionDef(
    override val name: String,
    override val variadicity: Variadicity = Variadicity.Single,
) extends OpInputDef
    with MayVariadicOpInputDef {}

/** Definition of a successor (control flow target block) for an operation.
  *
  * @param name
  *   The name of the successor.
  * @param variadicity
  *   Whether this is a single, variadic, or optional successor.
  */
case class SuccessorDef(
    override val name: String,
    override val variadicity: Variadicity = Variadicity.Single,
) extends OpInputDef
    with MayVariadicOpInputDef {}

/** Definition of a property (compile-time attribute) for an operation.
  *
  * @param name
  *   The name of the property.
  * @param tpe
  *   The expected type of the property.
  * @param variadicity
  *   Whether this is a single or optional property (variadic not supported).
  * @param constraint
  *   Optional type constraint on the property.
  */
case class OpPropertyDef(
    override val name: String,
    val tpe: Type[?],
    override val variadicity: Variadicity.Single.type |
      Variadicity.Optional.type = Variadicity.Single,
    val constraint: Option[Expr[ConstraintImpl[?]]] = None,
) extends OpInputDef
    with MayVariadicOpInputDef {}

/** Definition of a parameter for an attribute.
  *
  * @param name
  *   The name of the parameter.
  * @param tpe
  *   The expected type of the parameter.
  */
case class AttributeParamDef(
    val name: String,
    val tpe: Type[?],
) {}

/*≡≡=---=≡≡≡≡≡=---=≡≡*\
||   OPERATION DEF   ||
\*≡==----=≡≡≡=----==≡*/

/** Complete definition of an operation, derived from its ADT representation.
  * Contains all information needed to generate parsers, printers, and
  * verifiers.
  *
  * @param name
  *   The fully qualified operation name (e.g., "dialect.operation").
  * @param className
  *   The Scala class name for the operation.
  * @param operands
  *   Sequence of operand definitions.
  * @param results
  *   Sequence of result definitions.
  * @param regions
  *   Sequence of region definitions.
  * @param successors
  *   Sequence of successor definitions.
  * @param properties
  *   Sequence of property definitions.
  * @param assemblyFormat
  *   Optional custom assembly format specification.
  */
case class OperationDef(
    val name: String,
    val className: String,
    val operands: Seq[OperandDef] = Seq(),
    val results: Seq[ResultDef] = Seq(),
    val regions: Seq[RegionDef] = Seq(),
    val successors: Seq[SuccessorDef] = Seq(),
    val properties: Seq[OpPropertyDef] = Seq(),
    val assemblyFormat: Option[AssemblyFormatDirective] = None,
):

  /** Returns all input definitions (operands, results, regions, successors,
    * properties).
    */
  def allDefs =
    operands ++ results ++ regions ++ successors ++ properties

  /** Returns all input definitions with their indices. */
  def allDefsWithIndex =
    operands.zipWithIndex ++ results.zipWithIndex ++ regions.zipWithIndex ++
      successors.zipWithIndex ++ properties.zipWithIndex

  /** Returns true if this operation has more than one variadic operand. */
  def hasMultiVariadicOperands =
    operands.count(_.variadicity == Variadicity.Variadic) > 1

  /** Returns true if this operation has more than one variadic result. */
  def hasMultiVariadicResults =
    results.count(_.variadicity == Variadicity.Variadic) > 1

  /** Returns true if this operation has more than one variadic region. */
  def hasMultiVariadicRegions =
    regions.count(_.variadicity == Variadicity.Variadic) > 1

  /** Returns true if this operation has more than one variadic successor. */
  def hasMultiVariadicSuccessors =
    successors.count(_.variadicity == Variadicity.Variadic) > 1

/*≡≡=---=≡≡≡≡≡=---=≡≡*\
||   ATTRIBUTE DEF   ||
\*≡==----=≡≡≡=----==≡*/

/** Definition of a parametrized attribute, derived from its ADT representation.
  *
  * @param name
  *   The fully qualified attribute name (e.g., "dialect.attribute").
  * @param attributes
  *   Sequence of attribute parameter definitions.
  */
case class AttributeDef(
    val name: String,
    val attributes: Seq[AttributeParamDef] = Seq(),
):

  /** Returns attribute parameters with their indices. */
  def allDefsWithIndex =
    attributes.zipWithIndex

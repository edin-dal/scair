package scair.clair.macros

import scair.Printer
import scair.ir.*

import scala.compiletime.deferred

// ████████╗ ██████╗░ ░█████╗░ ██╗ ████████╗ ░██████╗
// ╚══██╔══╝ ██╔══██╗ ██╔══██╗ ██║ ╚══██╔══╝ ██╔════╝
// ░░░██║░░░ ██████╔╝ ███████║ ██║ ░░░██║░░░ ╚█████╗░
// ░░░██║░░░ ██╔══██╗ ██╔══██║ ██║ ░░░██║░░░ ░╚═══██╗
// ░░░██║░░░ ██║░░██║ ██║░░██║ ██║ ░░░██║░░░ ██████╔╝
// ░░░╚═╝░░░ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ╚═╝ ░░░╚═╝░░░ ╚═════╝░

object DerivedAttribute:

  inline given [T <: DerivedAttribute[?, ?]]: DerivedAttributeCompanion[T] =
    DerivedAttributeCompanion.derived[T]

transparent trait DerivedAttribute[name <: String, T <: Attribute]
    extends ParametrizedAttribute:

  this: T =>

  given companion: DerivedAttributeCompanion[T] = deferred
  override val name: String = companion.name

  override val parameters: Seq[Attribute | Seq[Attribute]] =
    companion.parameters(this)

trait AssemblyFormat[format <: String]

object DerivedOperation:

  inline given [T <: DerivedOperation[?, ?]]: DerivedOperationCompanion[T] =
    DerivedOperationCompanion.derived[T]

transparent trait DerivedOperation[name <: String, T] extends Operation:

  this: T =>

  given companion: DerivedOperationCompanion[T] = deferred

  override def updated(
      operands: Seq[Value[Attribute]],
      successors: Seq[Block],
      results: Seq[Result[Attribute]],
      regions: Seq[Region],
      properties: Map[String, Attribute],
      attributes: DictType[String, Attribute]
  ) =
    companion(
      operands = operands,
      successors = successors,
      results = results,
      regions = detached_regions,
      properties = properties,
      attributes = attributes
    )

  def name: String = companion.name
  def operands: Seq[Value[Attribute]] = companion.operands(this)
  def successors: Seq[Block] = companion.successors(this)
  def results: Seq[Result[Attribute]] = companion.results(this)
  def regions: Seq[Region] = companion.regions(this)
  def properties: Map[String, Attribute] = companion.properties(this)

  override def custom_print(p: Printer)(using indentLevel: Int): Unit =
    companion.custom_print(this, p)

  override def verify(): Either[String, Operation] =
    super
      .verify()
      .flatMap(_ => constraint_verify())

  def constraint_verify(): Either[String, Operation] =
    companion.constraint_verify(this)

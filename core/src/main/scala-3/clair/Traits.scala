package scair.clair.macros

import scair.Printer
import scair.ir.*

// ████████╗ ██████╗░ ░█████╗░ ██╗ ████████╗ ░██████╗
// ╚══██╔══╝ ██╔══██╗ ██╔══██╗ ██║ ╚══██╔══╝ ██╔════╝
// ░░░██║░░░ ██████╔╝ ███████║ ██║ ░░░██║░░░ ╚█████╗░
// ░░░██║░░░ ██╔══██╗ ██╔══██║ ██║ ░░░██║░░░ ░╚═══██╗
// ░░░██║░░░ ██║░░██║ ██║░░██║ ██║ ░░░██║░░░ ██████╔╝
// ░░░╚═╝░░░ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ╚═╝ ░░░╚═╝░░░ ╚═════╝░

transparent trait DerivedAttribute[name <: String, T <: Attribute](using
    private final val comp: DerivedAttributeCompanion[T]
) extends ParametrizedAttribute:

  this: T =>

  override val name: String = comp.name

  override val parameters: Seq[Attribute | Seq[Attribute]] =
    comp.parameters(this)

trait AssemblyFormat[format <: String]

abstract class DerivedOperation[name <: String, T <: Operation](using
    private final val comp: DerivedOperationCompanion[T]
) extends Operation:

  this: T =>

  override def updated(
      operands: Seq[Value[Attribute]],
      successors: Seq[Block],
      results: Seq[Result[Attribute]],
      regions: Seq[Region],
      properties: Map[String, Attribute],
      attributes: DictType[String, Attribute]
  ) =
    comp(
      operands = operands,
      successors = successors,
      results = results,
      regions = detached_regions,
      properties = properties,
      attributes = attributes
    )

  def name: String = comp.name
  def operands: Seq[Value[Attribute]] = comp.operands(this)
  def successors: Seq[Block] = comp.successors(this)
  def results: Seq[Result[Attribute]] = comp.results(this)
  def regions: Seq[Region] = comp.regions(this)
  def properties: Map[String, Attribute] = comp.properties(this)

  override def custom_print(p: Printer)(using indentLevel: Int): Unit =
    comp.custom_print(this, p)

  override def verify(): Either[String, Operation] =
    super
      .verify()
      .flatMap(_ => constraint_verify())

  def constraint_verify(): Either[String, Operation] =
    comp.constraint_verify(this)

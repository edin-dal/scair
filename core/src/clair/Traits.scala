package scair.clair

import scair.Printer
import scair.ir.*
import scair.utils.*

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

  def name: String = comp.name
  def operands: Seq[Value[Attribute]] = comp.operands(this)
  def successors: Seq[Block] = comp.successors(this)
  def results: Seq[Result[Attribute]] = comp.results(this)
  def regions: Seq[Region] = comp.regions(this)
  def properties: Map[String, Attribute] = comp.properties(this)

  override def customPrint(p: Printer): Unit =
    comp.customPrint(this, p)

  override def verify(): OK[Operation] =
    super.verify().flatMap(_ => constraintVerify())

  def constraintVerify(): OK[Operation] =
    comp.constraintVerify(this)

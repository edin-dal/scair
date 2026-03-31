package scair.clair

import scair.Printer
import scair.ir.*
import scair.utils.*

import scala.compiletime.deferred

// ████████╗ ██████╗░ ░█████╗░ ██╗ ████████╗ ░██████╗
// ╚══██╔══╝ ██╔══██╗ ██╔══██╗ ██║ ╚══██╔══╝ ██╔════╝
// ░░░██║░░░ ██████╔╝ ███████║ ██║ ░░░██║░░░ ╚█████╗░
// ░░░██║░░░ ██╔══██╗ ██╔══██║ ██║ ░░░██║░░░ ░╚═══██╗
// ░░░██║░░░ ██║░░██║ ██║░░██║ ██║ ░░░██║░░░ ██████╔╝
// ░░░╚═╝░░░ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ╚═╝ ░░░╚═╝░░░ ╚═════╝░

transparent trait DerivedAttribute[name <: String]
    extends ParametrizedAttribute:

  // This enables summoning the right instance without an explicit type parameter
  protected final given defs
      : AttrDefs[? >: this.type <: DerivedAttribute[name]] = deferred

  override val name: String = defs.name

  override val parameters: Seq[Attribute | Seq[Attribute]] =
    defs.parameters(this)

trait AssemblyFormat[format <: String]

transparent trait DerivedOperation[name <: String] extends Operation:

  // This enables summoning the right instance without an explicit type parameter
  protected final given defs: OpDefs[? >: this.type <: DerivedOperation[name]] =
    deferred

  override def updated(
      operands: Seq[Value[Attribute]],
      successors: Seq[Block],
      results: Seq[Result[Attribute]],
      regions: Seq[Region],
      properties: Map[String, Attribute],
      attributes: DictType[String, Attribute],
  ) =
    defs(
      operands = operands,
      successors = successors,
      results = results,
      regions = detachedRegions,
      properties = properties,
      attributes = attributes,
    )

  def name: String = defs.name
  def operands: Seq[Value[Attribute]] = defs.operands(this)
  def successors: Seq[Block] = defs.successors(this)
  def results: Seq[Result[Attribute]] = defs.results(this)
  def regions: Seq[Region] = defs.regions(this)
  def properties: Map[String, Attribute] = defs.properties(this)

  override def customPrint(p: Printer): Unit =
    defs.customPrint(this, p)

  override def verify(): OK[Operation] =
    super.verify().flatMap(_ => constraintVerify())

  def constraintVerify(): OK[Operation] =
    defs.constraintVerify(this)

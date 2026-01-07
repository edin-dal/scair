package scair.ir

import fastparse.P
import scair.Printer
import scair.parse.Parser
import scair.transformations.RewritePattern
import scair.utils.*
import scair.utils.IntrusiveNode

import scala.collection.mutable
import scala.collection.mutable.LinkedHashMap

//
// ░█████╗░ ██████╗░ ███████╗ ██████╗░ ░█████╗░ ████████╗ ██╗ ░█████╗░ ███╗░░██╗
// ██╔══██╗ ██╔══██╗ ██╔════╝ ██╔══██╗ ██╔══██╗ ╚══██╔══╝ ██║ ██╔══██╗ ████╗░██║
// ██║░░██║ ██████╔╝ █████╗░░ ██████╔╝ ███████║ ░░░██║░░░ ██║ ██║░░██║ ██╔██╗██║
// ██║░░██║ ██╔═══╝░ ██╔══╝░░ ██╔══██╗ ██╔══██║ ░░░██║░░░ ██║ ██║░░██║ ██║╚████║
// ╚█████╔╝ ██║░░░░░ ███████╗ ██║░░██║ ██║░░██║ ░░░██║░░░ ██║ ╚█████╔╝ ██║░╚███║
// ░╚════╝░ ╚═╝░░░░░ ╚══════╝ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ░░░╚═╝░░░ ╚═╝ ░╚════╝░ ╚═╝░░╚══╝
//

/*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
||    MLIR OPERATIONS    ||
\*≡==---==≡≡≡≡≡≡≡==---==≡*/
trait IRNode:
  def parent: Option[IRNode]

  final def isAncestor(other: IRNode): Boolean =
    other.parent match
      case Some(parent) if parent == this => true
      case Some(parent)                   => isAncestor(parent)
      case None                           => false
      case null                           => false

  def deepCopy(using
      blockMapper: mutable.Map[Block, Block] = mutable.Map.empty,
      valueMapper: mutable.Map[Value[Attribute], Value[Attribute]] = mutable.Map
        .empty,
  ): IRNode

trait Operation extends IRNode with IntrusiveNode[Operation]:

  final override def parent = containerBlock

  final override def deepCopy(using
      blockMapper: mutable.Map[Block, Block] = mutable.Map.empty,
      valueMapper: mutable.Map[Value[Attribute], Value[Attribute]] = mutable.Map
        .empty,
  ): Operation =
    val newResults = results.map(_.copy())
    valueMapper addAll (results zip newResults)
    updated(
      results = newResults.asInstanceOf[Seq[Result[Attribute]]],
      operands = operands.map(o => valueMapper.getOrElse(o, o)),
      successors = successors.map(b => blockMapper.getOrElseUpdate(b, b)),
      regions = regions.map(_.deepCopy),
      attributes = LinkedHashMap.from(attributes),
    )

  regions.foreach(attachRegion)

  results.foreach(r =>
    // if r.owner != None then
    //   throw new Exception(
    //     s"Result '${r.typ}' already has an owner: ${r.owner.get}"
    //   )
    // else
    r.owner = Some(this)
  )

  def name: String

  def updated(
      operands: Seq[Value[Attribute]] = operands,
      successors: Seq[Block] = successors,
      results: Seq[Result[Attribute]] = results.map(_.typ).map(Result(_)),
      regions: Seq[Region] = detachedRegions,
      properties: Map[String, Attribute] = properties,
      attributes: DictType[String, Attribute] = attributes,
  ): Operation

  def operands: Seq[Value[Attribute]]
  def successors: Seq[Block]
  def results: Seq[Result[Attribute]]
  def regions: Seq[Region]
  final def detachedRegions = regions.map(_.detached)
  def properties: Map[String, Attribute]
  val attributes: DictType[String, Attribute] = DictType.empty
  var containerBlock: Option[Block] = None
  def traitVerify(): OK[Operation] = OK(this)

  def customPrint(p: Printer)(using indentLevel: Int) =
    p.printGenericMLIROperation(this)

  def customVerify(): OK[Operation] = OK(this)

  def structured: OK[Operation] = regions
    .foldLeft[OK[Unit]](OK())((res, reg) => res.flatMap(_ => reg.structured))
    .map(_ => this)

  def verify(): OK[Operation] =
    results.foldLeft[OK[Unit]](OK())((res, result) =>
      res.flatMap(_ => result.verify())
    ).flatMap(_ =>
      regions.foldLeft[OK[Unit]](OK())((res, region) =>
        res.flatMap(_ => region.verify())
      )
    ).flatMap(_ =>
      properties.values.toSeq.foldLeft[OK[Unit]](OK())((res, prop) =>
        res.flatMap(_ => prop.customVerify())
      )
    ).flatMap(_ =>
      attributes.values.toSeq.foldLeft[OK[Unit]](OK())((res, attr) =>
        res.flatMap(_ => attr.customVerify())
      )
    ).flatMap(_ => traitVerify()).flatMap(_ => customVerify())

  final def dropAllReferences: Unit =
    containerBlock = None

  final def erase(safeErase: Boolean = true): Unit =
    if containerBlock != None then
      throw new Exception(
        "Operation should be first detached from its container block before erasure."
      )
    dropAllReferences
    if safeErase then for result <- results do result.erase()

  final def attachRegion(region: Region) =
    region.containerOperation match
      case Some(x) =>
        throw new Exception(
          s"""Can't attach a region already attached to an operation:
              ${Printer().print(region)(using 0)}"""
        )
      case None =>
        region.isAncestor(this) match
          case true =>
            throw new Exception(
              "Can't add a region to an operation that is contained within that region"
            )
          case false =>
            region.containerOperation = Some(this)

  final override def hashCode(): Int = System.identityHashCode(this)
  final override def equals(o: Any): Boolean = this eq o.asInstanceOf[Object]

object UnregisteredOperation:

  def apply(_name: String) =
    new OperationCompanion[UnregisteredOperation]:
      override def name = _name

      def apply(
          operands: Seq[Value[Attribute]] = Seq(),
          successors: Seq[Block] = Seq(),
          results: Seq[Result[Attribute]] = Seq(),
          regions: Seq[Region] = Seq(),
          properties: Map[String, Attribute] = Map.empty[String, Attribute],
          attributes: DictType[String, Attribute] = DictType
            .empty[String, Attribute],
      ): UnregisteredOperation =
        new UnregisteredOperation(
          name = _name,
          operands = operands,
          successors = successors,
          results = results,
          regions = regions,
          properties = properties,
          attributes = attributes,
        )

case class UnregisteredOperation private (
    override val name: String,
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
  ) =
    UnregisteredOperation(name)(
      operands = operands,
      successors = successors,
      results = results,
      regions = regions,
      properties = properties,
      attributes = attributes,
    )

trait OperationCompanion[O <: Operation]:
  def name: String

  def parse[$: P](resNames: Seq[String])(using Parser): P[O] =
    fastparse
      .Fail(
        s"No custom Parser implemented for Operation '$name'"
      )

  def apply(
      operands: Seq[Value[Attribute]] = Seq(),
      successors: Seq[Block] = Seq(),
      results: Seq[Result[Attribute]] = Seq(),
      regions: Seq[Region] = Seq(),
      properties: Map[String, Attribute] = Map.empty[String, Attribute],
      attributes: DictType[String, Attribute] = DictType
        .empty[String, Attribute],
  ): Operation

  def canonicalizationPatterns: Seq[RewritePattern] = Seq()
  export scair.parse.whitespace

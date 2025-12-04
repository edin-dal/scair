package scair.ir

import fastparse.P
import scair.Parser
import scair.Printer
import scair.transformations.RewritePattern
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

  final def is_ancestor(other: IRNode): Boolean =
    other.parent match
      case Some(parent) if parent == this => true
      case Some(parent)                   => is_ancestor(parent)
      case None                           => false
      case null                           => false

  def deepCopy(using
      blockMapper: mutable.Map[Block, Block] = mutable.Map.empty,
      valueMapper: mutable.Map[Value[Attribute], Value[Attribute]] =
        mutable.Map.empty
  ): IRNode

trait Operation extends IRNode with IntrusiveNode[Operation]:

  final override def parent = container_block

  final override def deepCopy(using
      blockMapper: mutable.Map[Block, Block] = mutable.Map.empty,
      valueMapper: mutable.Map[Value[Attribute], Value[Attribute]] =
        mutable.Map.empty
  ): Operation =
    val newResults = results.map(_.copy())
    valueMapper addAll (results zip newResults)
    updated(
      results = newResults.asInstanceOf[Seq[Result[Attribute]]],
      operands = operands.map(o => valueMapper.getOrElse(o, o)),
      successors = successors.map(b => blockMapper.getOrElseUpdate(b, b)),
      regions = regions.map(_.deepCopy),
      attributes = LinkedHashMap.from(attributes)
    )

  regions.foreach(attach_region)

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
      regions: Seq[Region] = detached_regions,
      properties: Map[String, Attribute] = properties,
      attributes: DictType[String, Attribute] = attributes
  ): Operation

  def operands: Seq[Value[Attribute]]
  def successors: Seq[Block]
  def results: Seq[Result[Attribute]]
  def regions: Seq[Region]
  final def detached_regions = regions.map(_.detached)
  def properties: Map[String, Attribute]
  val attributes: DictType[String, Attribute] = DictType.empty
  var container_block: Option[Block] = None
  def trait_verify(): Either[String, Operation] = Right(this)

  def custom_print(p: Printer)(using indentLevel: Int) =
    p.printGenericMLIROperation(this)

  def custom_verify(): Either[String, Operation] = Right(this)

  def structured: Either[String, Operation] = regions
    .foldLeft[Either[String, Unit]](Right(()))((res, reg) =>
      res.flatMap(_ => reg.structured)
    )
    .map(_ => this)

  def verify(): Either[String, Operation] =
    results
      .foldLeft[Either[String, Unit]](Right(()))((res, result) =>
        res.flatMap(_ => result.verify())
      )
      .flatMap(_ =>
        regions.foldLeft[Either[String, Unit]](Right(()))((res, region) =>
          res.flatMap(_ => region.verify())
        )
      )
      .flatMap(_ =>
        properties.values.toSeq.foldLeft[Either[String, Unit]](Right(()))(
          (res, prop) => res.flatMap(_ => prop.custom_verify())
        )
      )
      .flatMap(_ =>
        attributes.values.toSeq.foldLeft[Either[String, Unit]](Right(()))(
          (res, attr) => res.flatMap(_ => attr.custom_verify())
        )
      )
      .flatMap(_ => trait_verify())
      .flatMap(_ => custom_verify())

  final def drop_all_references: Unit =
    container_block = None

  final def erase(safe_erase: Boolean = true): Unit =
    if container_block != None then
      throw new Exception(
        "Operation should be first detached from its container block before erasure."
      )
    drop_all_references
    if safe_erase then for result <- results do result.erase()

  final def attach_region(region: Region) =
    region.container_operation match
      case Some(x) =>
        throw new Exception(
          s"""Can't attach a region already attached to an operation:
              ${Printer().print(region)(using 0)}"""
        )
      case None =>
        region.is_ancestor(this) match
          case true =>
            throw new Exception(
              "Can't add a region to an operation that is contained within that region"
            )
          case false =>
            region.container_operation = Some(this)

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
          attributes: DictType[String, Attribute] =
            DictType.empty[String, Attribute]
      ): UnregisteredOperation =
        new UnregisteredOperation(
          name = _name,
          operands = operands,
          successors = successors,
          results = results,
          regions = regions,
          properties = properties,
          attributes = attributes
        )

case class UnregisteredOperation private (
    override val name: String,
    override val operands: Seq[Value[Attribute]] = Seq(),
    override val successors: Seq[Block] = Seq(),
    override val results: Seq[Result[Attribute]] = Seq(),
    override val regions: Seq[Region] = Seq(),
    override val properties: Map[String, Attribute] =
      Map.empty[String, Attribute],
    override val attributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends Operation:

  override def updated(
      operands: Seq[Value[Attribute]] = operands,
      successors: Seq[Block] = successors,
      results: Seq[Result[Attribute]] = results.map(_.typ).map(Result(_)),
      regions: Seq[Region] = detached_regions,
      properties: Map[String, Attribute] = properties,
      attributes: DictType[String, Attribute] = attributes
  ) =
    UnregisteredOperation(name)(
      operands = operands,
      successors = successors,
      results = results,
      regions = regions,
      properties = properties,
      attributes = attributes
    )

trait OperationCompanion[O <: Operation]:
  def name: String

  def parse[$: P](parser: Parser, resNames: Seq[String]): P[O] =
    fastparse.Fail(
      s"No custom Parser implemented for Operation '${name}'"
    )

  def apply(
      operands: Seq[Value[Attribute]] = Seq(),
      successors: Seq[Block] = Seq(),
      results: Seq[Result[Attribute]] = Seq(),
      regions: Seq[Region] = Seq(),
      properties: Map[String, Attribute] = Map.empty[String, Attribute],
      attributes: DictType[String, Attribute] =
        DictType.empty[String, Attribute]
  ): Operation

  def canonicalizationPatterns: Seq[RewritePattern] = Seq()

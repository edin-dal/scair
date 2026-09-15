package scair.ir

import fastparse.P
import scair.collection.IntrusiveNode
import scair.helpers.foreachInline
import scair.parse.Parser
import scair.print.AssemblyPrinter
import scair.print.Printer
import scair.transformations.RewritePattern
import scair.utils.*

import scala.collection.mutable

//
// ░█████╗░ ██████╗░ ███████╗ ██████╗░ ░█████╗░ ████████╗ ██╗ ░█████╗░ ███╗░░██╗
// ██╔══██╗ ██╔══██╗ ██╔════╝ ██╔══██╗ ██╔══██╗ ╚══██╔══╝ ██║ ██╔══██╗ ████╗░██║
// ██║░░██║ ██████╔╝ █████╗░░ ██████╔╝ ███████║ ░░░██║░░░ ██║ ██║░░██║ ██╔██╗██║
// ██║░░██║ ██╔═══╝░ ██╔══╝░░ ██╔══██╗ ██╔══██║ ░░░██║░░░ ██║ ██║░░██║ ██║╚████║
// ╚█████╔╝ ██║░░░░░ ███████╗ ██║░░██║ ██║░░██║ ░░░██║░░░ ██║ ╚█████╔╝ ██║░╚███║
// ░╚════╝░ ╚═╝░░░░░ ╚══════╝ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ░░░╚═╝░░░ ╚═╝ ░╚════╝░ ╚═╝░░╚══╝
//

trait Operation extends IRNode with IntrusiveNode[Operation]:

  /*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
  ||   OPERATION INITIALIZATION   ||
  \*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

  var containerBlock: Option[Block] = None
  final override def parent = containerBlock
  var blockIndex = -1

  regions.foreach(attachRegion)

  results.foreach(r =>
    // if r.owner != None then
    //   throw new Exception(
    //     s"Result '${r.typ}' already has an owner: ${r.owner.get}"
    //   )
    // else
    r.owner = Some(this)
  )

  /*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
  ||   OPERATION GENERIC INTERFACE   ||
  \*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

  def name: String

  def operands: Seq[Value[Attribute]]
  def successors: Seq[Block]
  def results: Seq[Result[Attribute]]
  def regions: Seq[Region]
  def properties: Map[String, Attribute]

  /** The operation's discardable attributes. Immutable so that attribute-less
    * operations (the vast majority) share `Map.empty` and allocate nothing;
    * write with `op.attributes += k -> v`, `++=` or plain assignment.
    */
  var attributes: Map[String, Attribute] = Map.empty

  /** Sets this operation's attributes and returns it, for chaining in
    * constructors.
    */
  final def withAttributes(attributes: Map[String, Attribute]): this.type =
    this.attributes = attributes
    this

  final def detachedRegions = regions.map(_.detached)

  def customPrint(p: Printer) =
    p.printGenericMLIROperation(this)

  /*≡==--==≡≡≡≡≡≡≡==--=≡≡*\
  ||   OPERATION UTILS   ||
  \*≡==---==≡≡≡≡≡==---==≡*/

  /*
   * Return an error message wrapping this operation. Purposefully shadowing the Err
   * constructor in an Operation's body, to just automatically wrap the error message
   * with the operation that caused it, without having to explicitly pass 'this' every
   * time.
   */
  def Err(msg: String, obj: Option[AnyRef] = Some(this)) = scair.utils
    .Err(msg, obj)

  def updated(
      operands: Seq[Value[Attribute]] = operands,
      successors: Seq[Block] = successors,
      results: Seq[Result[Attribute]] = results.map(_.typ).map(Result(_)),
      regions: Seq[Region] = detachedRegions,
      properties: Map[String, Attribute] = properties,
      attributes: Map[String, Attribute] = attributes,
  ): Operation

  /*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
  ||   OPERATION TRANSFORMATIONS   ||
  \*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

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
              ${AssemblyPrinter().print(region)}"""
        )
      case None =>
        region.isAncestor(this) match
          case true =>
            throw new Exception(
              "Can't add a region to an operation that is contained within that region"
            )
          case false =>
            region.containerOperation = Some(this)

  /*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
  ||   OPERATION TRAVERSAL   ||
  \*≡==---==≡≡≡≡≡≡≡≡≡≡==---==≡*/

  /** Applies f to each region, inlined: no closure nor iterator. */
  inline final def forEachRegion(inline f: Region => Unit): Unit =
    regions.foreachInline(f)

  /** Applies f to each operand, inlined: no closure nor iterator. */
  inline final def forEachOperand(inline f: Value[Attribute] => Unit): Unit =
    operands.foreachInline(f)

  /** The operation directly containing this one, if any. */
  final def parentOp: Option[Operation] =
    containerBlock match
      case Some(block) =>
        block.containerRegion match
          case Some(region) => region.containerOperation
          case None         => None
      case None => None

  /** Walks this operation and all nested ones in pre-order. See WalkResult. */
  final def walk(f: Operation => WalkResult): WalkResult =
    f(this) match
      case WalkResult.Advance   => walkRegions(f)
      case WalkResult.Skip      => WalkResult.Advance
      case WalkResult.Interrupt => WalkResult.Interrupt

  /** Walks all nested operations then this one, in post-order. See WalkResult.
    */
  final def walkPostOrder(f: Operation => WalkResult): WalkResult =
    walkRegionsPostOrder(f) match
      case WalkResult.Interrupt => WalkResult.Interrupt
      case _                    =>
        f(this) match
          case WalkResult.Interrupt => WalkResult.Interrupt
          case _                    => WalkResult.Advance

  /** Walks this operation and all nested ones in pre-order, applying f. */
  inline final def walkAll(inline f: Operation => Unit): Unit =
    walk(op =>
      f(op)
      WalkResult.Advance
    )

  private def walkRegions(f: Operation => WalkResult): WalkResult =
    var result = WalkResult.Advance
    forEachRegion(region =>
      if result ne WalkResult.Interrupt then result = region.walk(f)
    )
    result

  private def walkRegionsPostOrder(f: Operation => WalkResult): WalkResult =
    var result = WalkResult.Advance
    forEachRegion(region =>
      if result ne WalkResult.Interrupt then result = region.walkPostOrder(f)
    )
    result

  /*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
  ||   OPERATION STRUCTURING   ||
  \*≡==---==≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

  override def recomputeOpOrder(): Unit =
    forEachRegion(_.recomputeOpOrder())

  def traitVerify(): OK[Operation] = OK(this)

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

  /*≡==--==≡≡≡≡≡≡==--=≡≡*\
  ||   OBJECT METHODS   ||
  \*≡==---==≡≡≡≡==---==≡*/

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
    )

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
          attributes: Map[String, Attribute] = Map.empty[String, Attribute],
      ): UnregisteredOperation =
        new UnregisteredOperation(
          name = _name,
          operands = operands,
          successors = successors,
          results = results,
          regions = regions,
          properties = properties,
        ).withAttributes(attributes)

case class UnregisteredOperation private (
    override val name: String,
    override val operands: Seq[Value[Attribute]],
    override val successors: Seq[Block],
    override val results: Seq[Result[Attribute]],
    override val regions: Seq[Region],
    override val properties: Map[String, Attribute],
) extends Operation:

  override def updated(
      operands: Seq[Value[Attribute]] = operands,
      successors: Seq[Block] = successors,
      results: Seq[Result[Attribute]] = results.map(_.typ).map(Result(_)),
      regions: Seq[Region] = detachedRegions,
      properties: Map[String, Attribute] = properties,
      attributes: Map[String, Attribute] = attributes,
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
      attributes: Map[String, Attribute] = Map.empty[String, Attribute],
  ): Operation

  def canonicalizationPatterns: Seq[RewritePattern] = Seq()
  export scair.parse.whitespace

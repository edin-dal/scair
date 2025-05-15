package scair.ir

import fastparse.P
import scair.Parser
import scair.Printer
// import scala.reflect.ClassTag

// ██╗ ██████╗░
// ██║ ██╔══██╗
// ██║ ██████╔╝
// ██║ ██╔══██╗
// ██║ ██║░░██║
// ╚═╝ ╚═╝░░╚═╝

/*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
||    MLIR OPERATIONS    ||
\*≡==---==≡≡≡≡≡≡≡==---==≡*/
trait IRNode {
  def parent: Option[IRNode]

  final def is_ancestor(other: IRNode): Boolean = {
    other.parent match {
      case Some(parent) if parent == this => true
      case Some(parent)                   => is_ancestor(parent)
      case None                           => false
      case null                           => false
    }
  }

}

trait Operation extends IRNode {

  final override def parent = container_block

  regions.foreach(attach_region)
  operands.zipWithIndex.foreach((o, i) => o.uses.addOne(Use(this, i)))

  def name: String

  def updated(
      operands: Seq[Value[Attribute]] = operands,
      successors: Seq[Block] = successors,
      results: Seq[Result[Attribute]] = results.map(_.typ).map(Result(_)),
      regions: Seq[Region] = regions,
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
  def trait_verify(): Either[Operation, String] = Left(this)

  def custom_print(p: Printer)(using indentLevel: Int) =
    p.printGenericMLIROperation(this)

  def custom_verify(): Either[Operation, String] = Left(this)

  def verify(): Either[Operation, String] = {

    lazy val verifyRes: Int => Either[Operation, String] = { (i: Int) =>
      if i == results.length then Left(this)
      else
        results(i).verify() match {
          case Left(v)  => verifyRes(i + 1)
          case Right(x) => Right(x)
        }
    }
    lazy val verifyReg: Int => Either[Operation, String] = { (i: Int) =>
      if i == regions.length then Left(this)
      else
        regions(i).verify() match {
          case Left(v)  => verifyReg(i + 1)
          case Right(x) => Right(x)
        }
    }
    lazy val verifyProp: Int => Either[Operation, String] = { (i: Int) =>
      if i == properties.size then Left(this)
      else
        properties.values.toSeq(i).custom_verify() match {
          case Left(v)  => verifyProp(i + 1)
          case Right(x) => Right(x)
        }
    }
    lazy val verifyAttrs: Int => Either[Operation, String] = { (i: Int) =>
      if i == attributes.size then Left(this)
      else
        attributes.values.toSeq(i).custom_verify() match {
          case Left(v)  => verifyAttrs(i + 1)
          case Right(x) => Right(x)
        }
    }
    verifyRes(0)
      .orElse(
        verifyReg(0)
      )
      .orElse(
        verifyProp(0)
      )
      .orElse(
        verifyAttrs(0)
      )
      .orElse(
        custom_verify()
      )
      .orElse(
        trait_verify()
      )
  }

  final def drop_all_references: Unit = {
    container_block = None
    operands.foreach(_.uses.filterInPlace(_.operation != this))
    for (region <- regions) region.drop_all_references
  }

  final def erase(safe_erase:Boolean = true): Unit = {
    if (container_block != None) then {
      throw new Exception(
        "Operation should be first detached from its container block before erasure."
      )
    }
    drop_all_references
    if (safe_erase) then {
      for (result <- results) {
        result.erase()
      }
    }
    
  }

  final def attach_region(region: Region) =
    region.container_operation match {
      case Some(x) =>
        throw new Exception(
          s"""Can't attach a region already attached to an operation:
              ${Printer().print(region)(using 0)}"""
        )
      case None =>
        region.is_ancestor(this) match {
          case true =>
            throw new Exception(
              "Can't add a region to an operation that is contained within that region"
            )
          case false =>
            region.container_operation = Some(this)
        }
    }

}

abstract class BaseOperation(
    val name: String,
    val operands: Seq[Value[Attribute]] = Seq(),
    val successors: Seq[Block] = Seq(),
    val results: Seq[Result[Attribute]] = Seq(),
    val regions: Seq[Region] = Seq(),
    val properties: Map[String, Attribute] = Map.empty[String, Attribute],
    override val attributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends Operation {

  // def companion : OperationCompanion

  def copy(
      operands: Seq[Value[Attribute]],
      successors: Seq[Block],
      results: Seq[Result[Attribute]],
      regions: Seq[Region],
      properties: Map[String, Attribute],
      attributes: DictType[String, Attribute]
  ): BaseOperation

  override def updated(
      operands: Seq[Value[Attribute]] = operands,
      successors: Seq[Block] = successors,
      results_types: Seq[Result[Attribute]] = results.map(_.typ).map(Result(_)),
      regions: Seq[Region] = regions,
      properties: Map[String, Attribute] = properties,
      attributes: DictType[String, Attribute] = attributes
  ) = {
    copy(
      operands = operands,
      successors = successors,
      results = results,
      regions = regions,
      properties = properties,
      attributes = attributes
    )
  }

  override def hashCode(): Int = {
    return 7 * 41 +
      this.operands.hashCode() +
      this.results.hashCode() +
      this.regions.hashCode() +
      this.properties.hashCode() +
      this.attributes.hashCode()
  }

  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }

}

case class UnregisteredOperation(
    override val name: String,
    override val operands: Seq[Value[Attribute]] = Seq(),
    override val successors: Seq[Block] = Seq(),
    override val results: Seq[Result[Attribute]] = Seq(),
    override val regions: Seq[Region] = Seq(),
    override val properties: Map[String, Attribute] =
      Map.empty[String, Attribute],
    override val attributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends BaseOperation(
      name = name,
      operands = operands,
      successors = successors,
      results = results.map(_.typ).map(Result(_)),
      regions = regions,
      properties = properties,
      attributes = attributes
    ) {

  override def copy(
      operands: Seq[Value[Attribute]],
      successors: Seq[Block],
      results: Seq[Result[Attribute]],
      regions: Seq[Region],
      properties: Map[String, Attribute],
      attributes: DictType[String, Attribute]
  ) = {
    UnregisteredOperation(
      name = name,
      operands = operands,
      successors = successors,
      results = results.map(_.typ).map(Result(_)),
      regions = regions.map(_.detached),
      properties = properties,
      attributes = attributes
    )
  }

}

trait OperationCompanion {
  def name: String

  def parse[$: P](parser: Parser): P[Operation] =
    throw new Exception(
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

}

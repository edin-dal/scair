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

trait Operation {
  def name: String

  def updated(
      operands: Seq[Value[Attribute]] = operands,
      successors: Seq[Block] = successors,
      results_types: Seq[Attribute] = results.map(_.typ),
      regions: Seq[Region] = regions,
      properties: DictType[String, Attribute] = properties,
      attributes: DictType[String, Attribute] = attributes
  ): Operation

  def operands: Seq[Value[Attribute]]
  def successors: Seq[Block]
  def results: Seq[Result[Attribute]]
  def regions: Seq[Region]
  def properties: DictType[String, Attribute]
  val attributes: DictType[String, Attribute] = DictType.empty
  var container_block: Option[Block] = None
  def trait_verify(): Unit = ()

  def custom_print(p: Printer): String =
    p.printGenericMLIROperation(this)

  def custom_verify(): Unit = ()

  final def verify(): Unit = {
    for (result <- results) result.verify()
    for (region <- regions) region.verify()
    for ((key, attr) <- properties) attr.custom_verify()
    for ((key, attr) <- attributes) attr.custom_verify()
    custom_verify()
    trait_verify()
  }

  final def drop_all_references: Unit = {
    container_block = None
    for ((idx, operand) <- (0 to operands.length) zip operands) {
      operand.remove_use(new Use(this, idx))
    }
    for (region <- regions) region.drop_all_references
  }

  final def is_ancestor(node: Block): Boolean = {
    val reg = node.container_region
    reg match {
      case Some(x) =>
        x.container_operation match {
          case Some(op) =>
            (op `equals` this) match {
              case true => true
              case false =>
                op.container_block match {
                  case None => false
                  case Some(block) =>
                    is_ancestor(block)
                }
            }
          case None => false
        }
      case None => false
    }
  }

  // TO-DO: think harder about the drop_refs - sounds fishy as per PR #45
  final def erase(drop_refs: Boolean = true): Unit = {
    if (container_block != None) then {
      throw new Exception(
        "Operation should be first detached from its container block before erasure."
      )
    }
    if (drop_refs) then drop_all_references

    for (result <- results) {
      result.erase()
    }
  }

}

abstract class BaseOperation(
    val name: String,
    val operands: Seq[Value[Attribute]] = Seq(),
    val successors: Seq[Block] = Seq(),
    val results_types: Seq[Attribute] = Seq(),
    val regions: Seq[Region] = Seq(),
    val properties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val attributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends Operation {

  // def companion : OperationCompanion

  def copy(
      operands: Seq[Value[Attribute]],
      successors: Seq[Block],
      results_types: Seq[Attribute],
      regions: Seq[Region],
      properties: DictType[String, Attribute],
      attributes: DictType[String, Attribute]
  ): BaseOperation

  override def updated(
      operands: Seq[Value[Attribute]] = operands,
      successors: Seq[Block] = successors,
      results_types: Seq[Attribute] = results.map(_.typ),
      regions: Seq[Region] = regions,
      properties: DictType[String, Attribute] = properties,
      attributes: DictType[String, Attribute] = attributes
  ) = {
    copy(
      operands = operands,
      successors = successors,
      results_types = results_types,
      regions = regions,
      properties = properties,
      attributes = attributes
    )
  }

  val results: Seq[Result[Attribute]] = results_types.map(Result(_))

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
    override val results_types: Seq[Attribute] = Seq(),
    override val regions: Seq[Region] = Seq(),
    override val properties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val attributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends BaseOperation(
      name = name,
      operands = operands,
      successors = successors,
      results_types = results_types,
      regions = regions,
      properties = properties,
      attributes = attributes
    ) {

  override def copy(
      operands: Seq[Value[Attribute]],
      successors: Seq[Block],
      results_types: Seq[Attribute],
      regions: Seq[Region],
      properties: DictType[String, Attribute],
      attributes: DictType[String, Attribute]
  ) = {
    UnregisteredOperation(
      name = name,
      operands = operands,
      successors = successors,
      results_types = results_types,
      regions = regions,
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
      results_types: Seq[Attribute] = Seq(),
      regions: Seq[Region] = Seq(),
      properties: DictType[String, Attribute] =
        DictType.empty[String, Attribute],
      attributes: DictType[String, Attribute] =
        DictType.empty[String, Attribute]
  ): Operation

}

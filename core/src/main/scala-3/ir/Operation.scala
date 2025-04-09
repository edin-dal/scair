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
  def operands: ListType[Value[Attribute]]
  def successors: ListType[Block]
  def results: ListType[Result[Attribute]]
  def regions: ListType[Region]
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
    val operands: ListType[Value[Attribute]] = ListType(),
    val successors: ListType[Block] = ListType(),
    results_types: ListType[Attribute] = ListType(),
    val regions: ListType[Region] = ListType(),
    val properties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val attributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends Operation {

  val results: ListType[Result[Attribute]] = results_types.map(Result(_))

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
    override val operands: ListType[Value[Attribute]] = ListType(),
    override val successors: ListType[Block] = ListType(),
    results_types: ListType[Attribute] = ListType(),
    override val regions: ListType[Region] = ListType(),
    override val properties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val attributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends BaseOperation(
      name = name,
      operands,
      successors,
      results_types,
      regions,
      properties,
      attributes
    )

trait OperationCompanion {
  def name: String

  def parse[$: P](parser: Parser): P[Operation] =
    throw new Exception(
      s"No custom Parser implemented for Operation '${name}'"
    )

  def apply(
      operands: ListType[Value[Attribute]] = ListType(),
      successors: ListType[Block] = ListType(),
      results_types: ListType[Attribute] = ListType(),
      regions: ListType[Region] = ListType(),
      properties: DictType[String, Attribute] =
        DictType.empty[String, Attribute],
      attributes: DictType[String, Attribute] =
        DictType.empty[String, Attribute]
  ): Operation

}

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

class MLIRName[name <: String]

sealed abstract class Operation()

class UnverifiedOp[T](
    name: String,
    operands: ListType[Value[Attribute]] = ListType(),
    successors: ListType[Block] = ListType(),
    results_types: ListType[Attribute] = ListType(),
    regions: ListType[Region] = ListType(),
    dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends MLIROperation(
      name = name,
      operands,
      successors,
      results_types,
      regions,
      dictionaryProperties,
      dictionaryAttributes
    )

/*≡==--==≡≡≡≡==--=≡≡*\
||    MLIR REALM    ||
\*≡==---==≡≡==---==≡*/

object MLIRRealm {}

trait MLIRRealm[T]() {

  def constructUnverifiedOp(
      operands: ListType[Value[Attribute]] = ListType(),
      successors: ListType[Block] = ListType(),
      results_types: ListType[Attribute] = ListType(),
      regions: ListType[Region] = ListType(),
      dictionaryProperties: DictType[String, Attribute] =
        DictType.empty[String, Attribute],
      dictionaryAttributes: DictType[String, Attribute] =
        DictType.empty[String, Attribute]
  ): UnverifiedOp[T]

  def unverify(op: T): UnverifiedOp[T]

  def verify(op: UnverifiedOp[T]): T

}

/*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
||    MLIR OPERATIONS    ||
\*≡==---==≡≡≡≡≡≡≡==---==≡*/

sealed abstract class MLIROperation(
    val name: String,
    val operands: ListType[Value[Attribute]] = ListType(),
    val successors: ListType[Block] = ListType(),
    results_types: ListType[Attribute] = ListType(),
    val regions: ListType[Region] = ListType(),
    val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends Operation,
      OpTrait {

  val results: ListType[Result[Attribute]] = results_types.map(Result(_))
  def op: MLIROperation = this

  var container_block: Option[Block] = None

  def is_ancestor(node: Block): Boolean = {
    val reg = node.container_region
    reg match {
      case Some(x) =>
        x.container_operation match {
          case Some(op) =>
            (op equals this) match {
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

  def drop_all_references: Unit = {
    container_block = None
    for ((idx, operand) <- (0 to operands.length) zip operands) {
      operand.remove_use(new Use(this, idx))
    }
    for (region <- regions) region.drop_all_references
  }

  // TO-DO: think harder about the drop_refs - sounds fishy as per PR #45
  def erase(drop_refs: Boolean = true): Unit = {
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

  def custom_verify(): Unit = ()

  final def verify(): Unit = {
    for (result <- results) result.verify()
    for (region <- regions) region.verify()
    for ((key, attr) <- dictionaryProperties) attr.custom_verify()
    for ((key, attr) <- dictionaryAttributes) attr.custom_verify()
    custom_verify()
    trait_verify()
  }

  def custom_print(p: Printer): String =
    p.printGenericMLIROperation(this)

  final def print(printer: Printer): String = {
    printer.printOperation(this)
  }

  override def hashCode(): Int = {
    return 7 * 41 +
      this.operands.hashCode() +
      this.results.hashCode() +
      this.regions.hashCode() +
      this.dictionaryProperties.hashCode() +
      this.dictionaryAttributes.hashCode()
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
    override val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends MLIROperation(
      name = name,
      operands,
      successors,
      results_types,
      regions,
      dictionaryProperties,
      dictionaryAttributes
    )

class RegisteredOperation(
    name: String,
    operands: ListType[Value[Attribute]] = ListType(),
    successors: ListType[Block] = ListType(),
    results_types: ListType[Attribute] = ListType(),
    regions: ListType[Region] = ListType(),
    dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends MLIROperation(
      name = name,
      operands,
      successors,
      results_types,
      regions,
      dictionaryProperties,
      dictionaryAttributes
    )

trait MLIROperationObject {
  def name: String

  def parse[$: P](parser: Parser): P[MLIROperation] =
    throw new Exception(
      s"No custom Parser implemented for Operation '${name}'"
    )

  type FactoryType = (
      ListType[Value[Attribute]] /* = operands */,
      ListType[Block] /* = successors */,
      ListType[Attribute] /* = results */,
      ListType[Region] /* = regions */,
      DictType[String, Attribute], /* = dictProps */
      DictType[String, Attribute] /* = dictAttrs */
  ) => MLIROperation

  def factory: FactoryType

  final def constructOp(
      operands: ListType[Value[Attribute]] = ListType(),
      successors: ListType[Block] = ListType(),
      results_types: ListType[Attribute] = ListType(),
      regions: ListType[Region] = ListType(),
      dictionaryProperties: DictType[String, Attribute] =
        DictType.empty[String, Attribute],
      dictionaryAttributes: DictType[String, Attribute] =
        DictType.empty[String, Attribute]
  ): MLIROperation = factory(
    operands,
    successors,
    results_types,
    regions,
    dictionaryProperties,
    dictionaryAttributes
  )

}

trait MLIRTraitI[T] {

  def getName: String

  def constructUnverifiedOp(
      operands: ListType[Value[Attribute]] = ListType(),
      successors: ListType[scair.ir.Block] = ListType(),
      results_types: ListType[Attribute] = ListType(),
      regions: ListType[Region] = ListType(),
      dictionaryProperties: DictType[String, Attribute] =
        DictType.empty[String, Attribute],
      dictionaryAttributes: DictType[String, Attribute] =
        DictType.empty[String, Attribute]
  ): UnverifiedOp[T]

  extension (adtOp: T) def MLIRTrait = this

  def unverify(adtOp: T): UnverifiedOp[T]

  def verify(unverOp: UnverifiedOp[T]): T

}

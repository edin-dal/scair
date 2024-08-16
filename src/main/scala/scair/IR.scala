package scair
import scala.collection.mutable.{Map, LinkedHashMap, ListBuffer}
import scair.Parser._
import fastparse._
import ListTypeExtensions.updateOperandsAndUses

// ==---------== //
// =---UTILS---= //
// ==---------== //

val DictType = LinkedHashMap
type DictType[A, B] = LinkedHashMap[A, B]

val ListType = ListBuffer
type ListType[A] = ListBuffer[A]

object ListTypeExtensions {
  extension(lt: ListType[Value[Attribute]]) {
    def updateOperandsAndUses(use: Use, newValue: Value[Attribute]): Unit = {
      newValue.uses += use
      lt.update(use.index, newValue)
    }
  }
}

// ==----------== //
// =-ATTRIBUTES-= //
// ==----------== //

sealed trait Attribute {
  def name: String
  def prefix: String = "#"
  def verify(): Unit = ()
}

trait TypeAttribute extends Attribute {
  override def prefix: String = "!"
}

abstract class ParametrizedAttribute(
    override val name: String,
    val parameters: Seq[Attribute] = Seq()
) extends Attribute {
  override def toString =
    s"${prefix}${name}<${parameters.map(x => x.toString).mkString(", ")}>"
}

abstract class DataAttribute[D](
    override val name: String,
    val data: D
) extends Attribute {
  override def toString = data.toString
}

// ==----------== //
// =---VALUES---= //
// ==----------== //

// TO-DO: perhaps a linked list of a use to other uses within an operation
//        for faster use retrieval and index update
class Use(val operation: Operation, val index: Int) {
  override def equals(o: Any): Boolean = o match {
    case use: Use =>
      (operation eq use.operation) && (index eq use.index)
    case _ => super.equals(o)
  }
}

object Value {
  def apply[T <: Attribute](typ: T): Value[T] = new Value(typ)
  def unapply[T <: Attribute](value: Value[T]): Option[T] = Some(value.typ)
}

class Value[T <: Attribute](
    var typ: T
) {

  var uses: ListType[Use] = ListType()

  def remove_use(use: Use): Unit = {
    val usesLengthBefore = uses.length
    uses -= use
    if (usesLengthBefore == uses.length) then {
      throw new Exception("Use to be removed was not in the Use list.")
    }
  }

  def replace_by(newValue: Value[Attribute]): Unit = {
    for (use <- uses) {
      use.operation.operands.updateOperandsAndUses(use, newValue)
    }
    uses = ListType()
  }

  def erase(): Unit = {
    if (uses.length != 0) then
    throw new Exception(
      "Attempting to erase a Value that has uses in other operations."
    )
  }

  def verify(): Unit = typ.verify()

  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }
}

// ==----------== //
// =---BLOCKS---= //
// ==----------== //

case class Block(
    operations: ListType[Operation] = ListType(),
    arguments: ListType[Value[Attribute]] = ListType()
) {

  var container_region: Option[Region] = None

  def drop_all_references: Unit = {
    container_region = None
    for (op <- operations) op.drop_all_references
  }

  def detach_op(op: Operation): Operation = {
    (op.container_block equals Some(this)) match {
      case true =>
        op.container_block = None
        operations -= op
        return op
      case false =>
        throw new Exception(
          "Operation can only be detached from a block in which it is contained."
        )
    }
  }

  def erase_op(op: Operation) = {
    detach_op(op)
    op.erase()
  }

  def verify(): Unit = {
    for (op <- operations) op.verify()
    for (arg <- arguments) arg.verify()
  }

  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }
}

// ==-----------== //
// =---REGIONS---= //
// ==-----------== //

case class Region(
    blocks: Seq[Block]
) {

  var container_operation: Option[Operation] = None

  def drop_all_references: Unit = {
    container_operation = None
    for (block <- blocks) block.drop_all_references
  }

  def verify(): Unit = {
    for (block <- blocks) block.verify()
  }
  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }
}

// ==----------== //
// =-OPERATIONS-= //
// ==----------== //

sealed abstract class Operation(
    val name: String,
    val operands: ListType[Value[Attribute]] = ListType(),
    val successors: ListType[Block] = ListType(),
    val results: ListType[Value[Attribute]] = ListType(),
    val regions: ListType[Region] = ListType(),
    val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) {

  var container_block: Option[Block] = None

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
    for ((key, attr) <- dictionaryProperties) attr.verify()
    for ((key, attr) <- dictionaryAttributes) attr.verify()
    custom_verify()
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

final case class UnregisteredOperation(
    override val name: String,
    override val operands: ListType[Value[Attribute]] = ListType(),
    override val successors: ListType[Block] = ListType(),
    override val results: ListType[Value[Attribute]] = ListType(),
    override val regions: ListType[Region] = ListType(),
    override val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends Operation(name = name)

class RegisteredOperation(
    override val name: String,
    override val operands: ListType[Value[Attribute]] = ListType(),
    override val successors: ListType[Block] = ListType(),
    override val results: ListType[Value[Attribute]] = ListType(),
    override val regions: ListType[Region] = ListType(),
    override val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends Operation(name = name)

// ==----------== //
// =--DIALECTS--= //
// ==----------== //

trait DialectOperation {
  def name: String
  def parse[$: P](resNames: Seq[String], parser: Parser): P[Operation] =
    throw new Exception(
      s"No custom Parser implemented for Operation '${name}'"
    )
  type FactoryType = (
      ListType[Value[Attribute]] /* = operands */,
      ListType[Block] /* = successors */,
      ListType[Value[Attribute]] /* = results */,
      ListType[Region] /* = regions */,
      DictType[String, Attribute], /* = dictProps */
      DictType[String, Attribute] /* = dictAttrs */
  ) => Operation
  def factory: FactoryType
  final def constructOp(
      operands: ListType[Value[Attribute]] = ListType(),
      successors: ListType[Block] = ListType(),
      results: ListType[Value[Attribute]] = ListType(),
      regions: ListType[Region] = ListType(),
      dictionaryProperties: DictType[String, Attribute] =
        DictType.empty[String, Attribute],
      dictionaryAttributes: DictType[String, Attribute] =
        DictType.empty[String, Attribute]
  ): Operation = factory(
    operands,
    successors,
    results,
    regions,
    dictionaryProperties,
    dictionaryAttributes
  )
}

trait DialectAttribute {
  def name: String
  type FactoryType = (Seq[Attribute]) => Attribute
  def factory: FactoryType = ???
  def parser[$: P]: P[Seq[Attribute]] =
    P(("<" ~ Type.rep(sep = ",") ~ ">").?).map(_.getOrElse(Seq()))
  def parse[$: P]: P[Attribute] =
    parser.map(factory(_))
}

final case class Dialect(
    val operations: Seq[DialectOperation],
    val attributes: Seq[DialectAttribute]
) {}

object IR {}

/*

import scala.quoted.*

trait MyObject {
  def name: String
}

trait MyTrait {
  type ReturnType <: MyObject

  inline def parseReturn(name: String):
    ReturnType = ${ MyTrait.impl[ReturnType]('name) }
}

object MyTrait {
  def impl[T <: MyObject: Type]
  (name: Expr[String])(using Quotes): Expr[T] = {
    import quotes.reflect.*

    // Generate code to instantiate the ReturnType with String argument
    val returnTypeSymbol = TypeRepr.of[T].typeSymbol
    val instance = New(TypeIdent(returnTypeSymbol))
      .select(returnTypeSymbol.primaryConstructor)
      .appliedToArgs(name)
      .asExprOf[T]

    instance
  }
}

 */

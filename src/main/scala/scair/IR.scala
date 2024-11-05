package scair.ir

import scala.collection.mutable.{Map, LinkedHashMap, ListBuffer}
import scair.Parser._
import scair.exceptions.VerifyException
import fastparse._
import scair.{Parser, Printer}

// ==---------== //
// =---UTILS---= //
// ==---------== //

// TO-DO: export utils to a separate file

val DictType = LinkedHashMap
type DictType[A, B] = LinkedHashMap[A, B]

val ListType = ListBuffer
type ListType[A] = ListBuffer[A]

extension (dt: DictType[String, Attribute]) {
  def checkandget(
      key: String,
      op_name: String,
      expected_type: String
  ): Attribute = {
    dt.get(key) match {
      case Some(b) => b
      case None =>
        throw new Exception(
          s"Operation '${op_name}' must include an attribute named '${key}' of type '${}'"
        )
    }
  }
}

extension (lt: ListType[Value[Attribute]]) {
  def updateOperandsAndUses(use: Use, newValue: Value[Attribute]): Unit = {
    newValue.uses += use
    lt.update(use.index, newValue)
  }
}

// ==----------== //
// =-ATTRIBUTES-= //
// ==----------== //

sealed trait Attribute {
  def name: String
  def prefix: String = "#"
  def custom_verify(): Unit = ()
  def custom_print: String
}

trait TypeAttribute extends Attribute {
  override def prefix: String = "!"
}

// TODO: Think about this; probably not the best design
extension (x: Seq[Attribute] | Attribute)
  def custom_print: String = x match {
    case seq: Seq[Attribute] => seq.map(_.custom_print).mkString("[", ", ", "]")
    case attr: Attribute     => attr.custom_print
  }
abstract class ParametrizedAttribute(
    override val name: String,
    val parameters: Seq[Attribute | Seq[Attribute]] = Seq()
) extends Attribute {
  override def custom_print =
    s"${prefix}${name}<${parameters.map(x => x.custom_print).mkString(", ")}>"
  override def equals(attr: Any): Boolean = {
    attr match {
      case x: ParametrizedAttribute =>
        x.name == this.name &&
        x.getClass == this.getClass &&
        x.parameters.length == this.parameters.length &&
        (for ((i, j) <- x.parameters zip this.parameters)
          yield i == j).foldLeft(true)((i, j) => i && j)
      case _ => false
    }
  }
}

abstract class DataAttribute[D](
    override val name: String,
    val data: D
) extends Attribute {
  override def custom_print = data.toString
  override def equals(attr: Any): Boolean = {
    attr match {
      case x: DataAttribute[D] =>
        x.name == this.name &&
        x.getClass == this.getClass &&
        x.data == this.data
      case _ => false
    }
  }
}

// ==----------== //
// =---VALUES---= //
// ==----------== //

// TO-DO: perhaps a linked list of a use to other uses within an operation
//        for faster use retrieval and index update
case class Use(val operation: Operation, val index: Int) {
  override def equals(o: Any): Boolean = o match {
    case Use(op, idx) =>
      (operation eq op) && (index eq idx)
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

  def verify(): Unit = typ.custom_verify()

  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }
}

extension (seq: Seq[Value[Attribute]]) def typ: Seq[Attribute] = seq.map(_.typ)

type Operand[T <: Attribute] = Value[T]
type OpResult[T <: Attribute] = Value[T]

// ==----------== //
// =---BLOCKS---= //
// ==----------== //

case class Block(
    operations: ListType[Operation] = ListType(),
    arguments: ListType[Value[Attribute]] = ListType()
) {

  var container_region: Option[Region] = None

  private def attach_op(op: Operation): Unit = {
    op.container_block match {
      case Some(x) =>
        throw new Exception(
          "Can't attach an operation already attached to a block."
        )
      case None =>
        op.is_ancestor(this) match {
          case true =>
            throw new Exception(
              "Can't add an operation to a block that is contained within that operation"
            )
          case false =>
            op.container_block = Some(this)
        }
    }
  }

  def add_op(new_op: Operation): Unit = {
    val oplen = operations.length
    attach_op(new_op)
    operations.insertAll(oplen, ListType(new_op))
  }

  def add_ops(new_ops: Seq[Operation]): Unit = {
    val oplen = operations.length
    for (op <- new_ops) {
      attach_op(op)
    }
    operations.insertAll(oplen, ListType(new_ops: _*))
  }

  def insert_op_before(existing_op: Operation, new_op: Operation): Unit = {
    (existing_op.container_block equals Some(this)) match {
      case true =>
        attach_op(new_op)
        operations.insertAll(getIndexOf(existing_op), ListType(new_op))
      case false =>
        throw new Exception(
          "Can't insert the new operation into the block, as the operation that was " +
            "given as a point of reference does not exist in the current block."
        )
    }
  }

  def insert_ops_before(
      existing_op: Operation,
      new_ops: Seq[Operation]
  ): Unit = {
    (existing_op.container_block equals Some(this)) match {
      case true =>
        for (op <- new_ops) {
          attach_op(op)
        }
        operations.insertAll(getIndexOf(existing_op), ListType(new_ops: _*))
      case false =>
        throw new Exception(
          "Can't insert the new operation into the block, as the operation that was " +
            "given as a point of reference does not exist in the current block."
        )
    }
  }

  def insert_op_after(existing_op: Operation, new_op: Operation): Unit = {
    (existing_op.container_block equals Some(this)) match {
      case true =>
        attach_op(new_op)
        operations.insertAll(getIndexOf(existing_op) + 1, ListType(new_op))
      case false =>
        throw new Exception(
          "Can't insert the new operation into the block, as the operation that was " +
            "given as a point of reference does not exist in the current block."
        )
    }
  }

  def insert_ops_after(
      existing_op: Operation,
      new_ops: Seq[Operation]
  ): Unit = {
    (existing_op.container_block equals Some(this)) match {
      case true =>
        for (op <- new_ops) {
          attach_op(op)
        }
        operations.insertAll(getIndexOf(existing_op) + 1, ListType(new_ops: _*))
      case false =>
        throw new Exception(
          "Can't insert the new operation into the block, as the operation that was " +
            "given as a point of reference does not exist in the current block."
        )
    }
  }

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

  def getIndexOf(op: Operation): Int = {
    operations.lastIndexOf(op) match {
      case -1 => throw new Exception("Operation not present in the block.")
      case x  => x
    }

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

/*≡==--=≡≡≡=--=≡≡*\
||    OPTRAIT    ||
\*≡==---=≡=---==≡*/

abstract class OpTrait {
    def op: Operation
    def trait_verify(): Unit = ()
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
) extends OpTrait {

  def op: Operation = this

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
    throw new Exception(
      s"No custom Printer implemented for Operation '${name}'"
    )

  final def print(printer: Printer): String = {
    var results: Seq[String] = Seq()
    var resultsTypes: Seq[String] = Seq()

    for { res <- this.results } yield (
      results = results :+ printer.printValue(res),
      resultsTypes = resultsTypes :+ printer.printAttribute(res.typ)
    )

    val operationResults: String =
      if (this.results.length > 0)
        results.mkString(", ") + " = "
      else ""

    return operationResults + custom_print(printer)
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

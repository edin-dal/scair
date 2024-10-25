package scair.dialects.affine

import fastparse._
import scala.collection.immutable
import scala.collection.mutable

import scair.dialects.affine._
import scair.dialects.builtin.{
  AffineMapAttr,
  AffineSetAttr,
  IndexType,
  DenseIntOrFPElementsAttr,
  ArrayAttribute
}
import scair.exceptions.VerifyException
import scair.dialects.irdl._
import scair.Parser.{whitespace, ValueId, Type, E}
import scair.AttrParser.{Float32TypeP, Float64TypeP}
import scair.DictTypeExtenions.checkandget
import scair.{
  ListType,
  DictType,
  Operation,
  RegisteredOperation,
  Region,
  Block,
  Value,
  Attribute,
  TypeAttribute,
  ParametrizedAttribute,
  DataAttribute,
  DialectAttribute,
  DialectOperation,
  Dialect,
  Printer,
  Parser
}

// ░█████╗░ ███████╗ ███████╗ ██╗ ███╗░░██╗ ███████╗
// ██╔══██╗ ██╔════╝ ██╔════╝ ██║ ████╗░██║ ██╔════╝
// ███████║ █████╗░░ █████╗░░ ██║ ██╔██╗██║ █████╗░░
// ██╔══██║ ██╔══╝░░ ██╔══╝░░ ██║ ██║╚████║ ██╔══╝░░
// ██║░░██║ ██║░░░░░ ██║░░░░░ ██║ ██║░╚███║ ███████╗
// ╚═╝░░╚═╝ ╚═╝░░░░░ ╚═╝░░░░░ ╚═╝ ╚═╝░░╚══╝ ╚══════╝

// ░█████╗░ ██████╗░ ███████╗ ██████╗░ ░█████╗░ ████████╗ ██╗ ░█████╗░ ███╗░░██╗ ░██████╗
// ██╔══██╗ ██╔══██╗ ██╔════╝ ██╔══██╗ ██╔══██╗ ╚══██╔══╝ ██║ ██╔══██╗ ████╗░██║ ██╔════╝
// ██║░░██║ ██████╔╝ █████╗░░ ██████╔╝ ███████║ ░░░██║░░░ ██║ ██║░░██║ ██╔██╗██║ ╚█████╗░
// ██║░░██║ ██╔═══╝░ ██╔══╝░░ ██╔══██╗ ██╔══██║ ░░░██║░░░ ██║ ██║░░██║ ██║╚████║ ░╚═══██╗
// ╚█████╔╝ ██║░░░░░ ███████╗ ██║░░██║ ██║░░██║ ░░░██║░░░ ██║ ╚█████╔╝ ██║░╚███║ ██████╔╝
// ░╚════╝░ ╚═╝░░░░░ ╚══════╝ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ░░░╚═╝░░░ ╚═╝ ░╚════╝░ ╚═╝░░╚══╝ ╚═════╝░

/*≡==---==≡≡≡≡==---=≡≡*\
||      APPLY OP      ||
\*≡==----==≡≡==----==≡*/

object ApplyOp extends DialectOperation {
  override def name: String = "affine.apply"
  override def factory: FactoryType = ApplyOp.apply
}

case class ApplyOp(
    override val operands: ListType[Value[Attribute]] = ListType(),
    override val successors: ListType[Block] = ListType(),
    override val results: ListType[Value[Attribute]] = ListType(),
    override val regions: ListType[Region] = ListType(),
    override val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends RegisteredOperation(name = "affine.apply") {

  val index_check = BaseAttr[IndexType.type]()
  val map_check = BaseAttr[AffineMapAttr]()

  override def custom_verify(): Unit = (
    successors.length,
    results.length,
    regions.length,
    dictionaryProperties.size,
    dictionaryAttributes.size
  ) match {
    case (0, 1, 0, 0, 1) =>
      for (x <- operands) index_check.verify(x.typ, new ConstraintContext())
      index_check.verify(results(0).typ, new ConstraintContext())
      map_check.verify(
        dictionaryAttributes.checkandget("map", name, "affine_map"),
        new ConstraintContext()
      )
    case _ =>
      throw new VerifyException(
        "Apply Operation must only contain at least 1 operand and exaclty 1 result of 'index' type, " +
          "as well as attribute in a dictionary called 'map' of 'affine_map' type."
      )
  }
}

/*≡==---=≡≡≡≡=---=≡≡*\
||      FOR OP      ||
\*≡==----=≡≡=----==≡*/

object ForOp extends DialectOperation {
  override def name: String = "affine.for"
  override def factory: FactoryType = ForOp.apply
}

case class ForOp(
    override val operands: ListType[Value[Attribute]] = ListType(),
    override val successors: ListType[Block] = ListType(),
    override val results: ListType[Value[Attribute]] = ListType(),
    override val regions: ListType[Region] = ListType(),
    override val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends RegisteredOperation(name = "affine.for") {

  val index_check = BaseAttr[IndexType.type]()
  val map_check = BaseAttr[AffineMapAttr]()

  override def custom_verify(): Unit = (
    successors.length,
    regions.length,
    dictionaryProperties.size,
    dictionaryAttributes.size
  ) match {
    case (0, 1, 0, 3) =>
      for (x <- operands) index_check.verify(x.typ, new ConstraintContext())
      map_check.verify(
        dictionaryAttributes.checkandget("lowerBoundMap", name, "affine_map"),
        new ConstraintContext()
      )
      map_check.verify(
        dictionaryAttributes.checkandget("upperBoundMap", name, "affine_map"),
        new ConstraintContext()
      )
      index_check.verify(
        dictionaryAttributes.checkandget("step", name, "index"),
        new ConstraintContext()
      )
    case _ =>
      throw new VerifyException(
        "For Operation must only contain operands of type index, and 3 dictionary attributes."
      )
  }
}

/*≡==---==≡≡≡≡≡==---=≡≡*\
||     PARALLEL OP     ||
\*≡==----==≡≡≡==----==≡*/

object ParallelOp extends DialectOperation {
  override def name: String = "affine.parallel"
  override def factory: FactoryType = ParallelOp.apply
}

case class ParallelOp(
    override val operands: ListType[Value[Attribute]] = ListType(),
    override val successors: ListType[Block] = ListType(),
    override val results: ListType[Value[Attribute]] = ListType(),
    override val regions: ListType[Region] = ListType(),
    override val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends RegisteredOperation(name = "affine.parallel") {

  val index_check = BaseAttr[IndexType.type]()
  val map_check = BaseAttr[AffineMapAttr]()
  val array_check = BaseAttr[ArrayAttribute[Attribute]]()
  val dense_check = BaseAttr[DenseIntOrFPElementsAttr]()

  override def custom_verify(): Unit = (
    successors.length,
    regions.length,
    dictionaryProperties.size,
    dictionaryAttributes.size
  ) match {
    case (0, 1, 0, 6) =>
      for (x <- operands) index_check.verify(x.typ, new ConstraintContext())
      array_check.verify(
        dictionaryAttributes.checkandget("reductions", name, "array_attribute"),
        new ConstraintContext()
      )
      map_check.verify(
        dictionaryAttributes.checkandget("lowerBoundMap", name, "affine_map"),
        new ConstraintContext()
      )
      dense_check.verify(
        dictionaryAttributes
          .checkandget("lowerBoundGroups", name, "dense"),
        new ConstraintContext()
      )
      map_check.verify(
        dictionaryAttributes.checkandget("upperBoundMap", name, "affine_map"),
        new ConstraintContext()
      )
      dense_check.verify(
        dictionaryAttributes
          .checkandget("upperBoundGroups", name, "dense"),
        new ConstraintContext()
      )
      array_check.verify(
        dictionaryAttributes.checkandget("steps", name, "index"),
        new ConstraintContext()
      )
    case _ =>
      throw new VerifyException(
        "Parallel Operation must only contain operands of type index, and 6 dictionary attributes."
      )
  }
}

/*≡==--=≡≡≡=--=≡≡*\
||     IF OP     ||
\*≡==---=≡=---==≡*/

object IfOp extends DialectOperation {
  override def name: String = "affine.if"
  override def factory: FactoryType = IfOp.apply
}

case class IfOp(
    override val operands: ListType[Value[Attribute]] = ListType(),
    override val successors: ListType[Block] = ListType(),
    override val results: ListType[Value[Attribute]] = ListType(),
    override val regions: ListType[Region] = ListType(),
    override val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends RegisteredOperation(name = "affine.if") {

  override def custom_verify(): Unit = (
    successors.length,
    regions.length,
    dictionaryProperties.size,
    dictionaryAttributes.size
  ) match {
    case (0, 0, 0, 0) =>
    case _ =>
      throw new VerifyException(
        "If Operation must only contain only operands and results."
      )
  }
}

/*≡==--=≡≡≡≡=--=≡≡*\
||    STORE OP    ||
\*≡==--==≡≡==--==≡*/

// TODO: operands: any, memref & indices

object StoreOp extends DialectOperation {
  override def name: String = "affine.store"
  override def factory: FactoryType = StoreOp.apply
}

case class StoreOp(
    override val operands: ListType[Value[Attribute]] = ListType(),
    override val successors: ListType[Block] = ListType(),
    override val results: ListType[Value[Attribute]] = ListType(),
    override val regions: ListType[Region] = ListType(),
    override val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends RegisteredOperation(name = "affine.store") {

  val map_check = BaseAttr[AffineMapAttr]()

  override def custom_verify(): Unit = (
    successors.length,
    regions.length,
    dictionaryProperties.size,
    dictionaryAttributes.size
  ) match {
    case (0, 0, 0, 0) =>
      map_check.verify(
        dictionaryAttributes.checkandget("map", name, "affine_map"),
        new ConstraintContext()
      )
    case _ =>
      throw new VerifyException(
        "If Operation must only contain only operands and results."
      )
  }
}

/*≡==---=≡≡≡=---=≡≡*\
||     LOAD OP     ||
\*≡==----=≡=----==≡*/

// TODO: operands: memref & indices

object LoadOp extends DialectOperation {
  override def name: String = "affine.load"
  override def factory: FactoryType = LoadOp.apply
}

case class LoadOp(
    override val operands: ListType[Value[Attribute]] = ListType(),
    override val successors: ListType[Block] = ListType(),
    override val results: ListType[Value[Attribute]] = ListType(),
    override val regions: ListType[Region] = ListType(),
    override val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends RegisteredOperation(name = "affine.load") {

  val map_check = BaseAttr[AffineMapAttr]()

  override def custom_verify(): Unit = (
    successors.length,
    regions.length,
    results.length,
    dictionaryProperties.size,
    dictionaryAttributes.size
  ) match {
    case (0, 0, 1, 0, 0) =>
      map_check.verify(
        dictionaryAttributes.checkandget("map", name, "affine_map"),
        new ConstraintContext()
      )
    case _ =>
      throw new VerifyException(
        "If Operation must only contain only operands and results."
      )
  }
}

/*≡==--=≡≡≡≡=--=≡≡*\
||     MIN OP     ||
\*≡==---=≡≡=---==≡*/

object MinOp extends DialectOperation {
  override def name: String = "affine.min"
  override def factory: FactoryType = MinOp.apply
}

case class MinOp(
    override val operands: ListType[Value[Attribute]] = ListType(),
    override val successors: ListType[Block] = ListType(),
    override val results: ListType[Value[Attribute]] = ListType(),
    override val regions: ListType[Region] = ListType(),
    override val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends RegisteredOperation(name = "affine.min") {

  val map_check = BaseAttr[AffineMapAttr]()
  val index_check = BaseAttr[IndexType.type]()

  override def custom_verify(): Unit = (
    successors.length,
    regions.length,
    results.length,
    dictionaryProperties.size,
    dictionaryAttributes.size
  ) match {
    case (0, 0, 1, 0, 0) =>
      for (x <- operands) index_check.verify(x.typ, new ConstraintContext())
      map_check.verify(
        dictionaryAttributes.checkandget("map", name, "affine_map"),
        new ConstraintContext()
      )
      index_check.verify(results(0).typ, new ConstraintContext())

    case _ =>
      throw new VerifyException(
        "If Operation must only contain only operands and results."
      )
  }
}

/*≡==--=≡≡≡≡=--=≡≡*\
||    YIELD OP    ||
\*≡==---=≡≡=---==≡*/

object YieldOp extends DialectOperation {
  override def name: String = "affine.yield"
  override def factory: FactoryType = YieldOp.apply
}

case class YieldOp(
    override val operands: ListType[Value[Attribute]] = ListType(),
    override val successors: ListType[Block] = ListType(),
    override val results: ListType[Value[Attribute]] = ListType(),
    override val regions: ListType[Region] = ListType(),
    override val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends RegisteredOperation(name = "affine.yield") {

  override def custom_verify(): Unit = (
    successors.length,
    regions.length,
    dictionaryProperties.size,
    dictionaryAttributes.size
  ) match {
    case (0, 0, 0, 0) =>
    case _ =>
      throw new VerifyException(
        "If Operation must only contain only operands and results."
      )
  }
}

val Affine: Dialect =
  new Dialect(
    operations =
      Seq(ApplyOp, ForOp, ParallelOp, IfOp, StoreOp, LoadOp, MinOp, YieldOp),
    attributes = Seq()
  )

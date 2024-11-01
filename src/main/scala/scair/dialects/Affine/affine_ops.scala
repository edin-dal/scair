package scair.dialects.affine

import fastparse._
import scala.collection.immutable
import scala.collection.mutable

import scair.dialects.affine._
import scair.dialects.builtin.{
  AffineMapAttr,
  AffineSetAttr,
  IntegerAttr,
  IndexType,
  DenseArrayAttr,
  DenseIntOrFPElementsAttr,
  ArrayAttribute
}
import scair.exceptions.VerifyException
import scair.Parser.{whitespace, ValueId, Type, E}
import scair.AttrParser.{Float32TypeP, Float64TypeP}
import scair.ir._
import scair.scairdl.constraints.{
  IRDLConstraint,
  ConstraintContext,
  BaseAttr,
  ParametrizedBaseAttr,
  AnyAttr
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

/*≡==---=≡≡≡=---=≡≡*\
||   CONSTRAINTS   ||
\*≡==----=≡=----==≡*/

val array_check = ParametrizedBaseAttr[DenseArrayAttr, IntegerAttr]()
val index_check = BaseAttr[IndexType.type]()
val map_check = BaseAttr[AffineMapAttr]()
val dense_check = BaseAttr[DenseIntOrFPElementsAttr]()

case class SegmentedOpConstraint(
    op_name: String,
    valList: ListType[Value[Attribute]],
    constraints: Seq[IRDLConstraint]
) extends IRDLConstraint {

  override def verify(
      that_attr: Attribute,
      constraint_ctx: ConstraintContext
  ): Unit = {
    that_attr match {
      case x: DenseArrayAttr =>
        assert(x.data.size == constraints.size) // error for developer
        var start: Int = 0
        var idx = 0
        for (a <- x.parameters) a match {
          case a: IntegerAttr =>
            val a_val = a.value.value.toInt
            for (i <- start to (a_val - 1))
              constraints(idx).verify(valList(i).typ, constraint_ctx)
            start += a_val
            idx += 1
          case _ =>
            val errstr =
              s"Element ${that_attr.name} in the DenseArrayAttr is not of type IntegerAttr\n"
            throw new Exception(errstr)
        }
      case _ =>
        val errstr =
          s"${that_attr.name}'s class does not equal DenseArrayAttr\n"
        throw new Exception(errstr)
    }
  }

  override def toString =
    s"SegmentedOp[DenseArrayAttr, IntegerAttr]"
}

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

  val seg_check = SegmentedOpConstraint(
    name,
    operands,
    Seq(index_check, index_check, AnyAttr)
  )

  override def custom_verify(): Unit = (
    successors.length,
    regions.length,
    dictionaryProperties.size,
    dictionaryAttributes.size
  ) match {
    case (0, 1, 0, 4) =>
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
      seg_check.verify(
        dictionaryAttributes.checkandget("operandsSegmentSizes", name, "array"),
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

  val array1_check = BaseAttr[ArrayAttribute[Attribute]]()

  override def custom_verify(): Unit = (
    successors.length,
    regions.length,
    dictionaryProperties.size,
    dictionaryAttributes.size
  ) match {
    case (0, 1, 0, 6) =>
      for (x <- operands) index_check.verify(x.typ, new ConstraintContext())
      array1_check.verify(
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
      array1_check.verify(
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

  override def custom_verify(): Unit = (
    successors.length,
    regions.length,
    dictionaryProperties.size,
    dictionaryAttributes.size
  ) match {
    case (0, 0, 0, 1) =>
      map_check.verify(
        dictionaryAttributes.checkandget("map", name, "affine_map"),
        new ConstraintContext()
      )
    case _ =>
      throw new Exception(
        "Store Operation must only contain only operands."
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

  override def custom_verify(): Unit = (
    successors.length,
    regions.length,
    results.length,
    dictionaryProperties.size,
    dictionaryAttributes.size
  ) match {
    case (0, 0, 1, 0, 1) =>
      map_check.verify(
        dictionaryAttributes.checkandget("map", name, "affine_map"),
        new ConstraintContext()
      )
    case _ =>
      throw new VerifyException(
        "Load Operation must only contain only operands and a result."
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

  override def custom_verify(): Unit = (
    successors.length,
    regions.length,
    results.length,
    dictionaryProperties.size,
    dictionaryAttributes.size
  ) match {
    case (0, 0, 1, 0, 1) =>
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
    results.length,
    regions.length,
    dictionaryProperties.size,
    dictionaryAttributes.size
  ) match {
    case (0, 0, 0, 0, 0) =>
    case _ =>
      throw new VerifyException(
        "Yield Operation must only contain only operands."
      )
  }
}

val Affine: Dialect =
  new Dialect(
    operations =
      Seq(ApplyOp, ForOp, ParallelOp, IfOp, StoreOp, LoadOp, MinOp, YieldOp),
    attributes = Seq()
  )

package scair.dialects.builtin

import fastparse.*
import scair.Parser
import scair.Printer
import scair.core.macros.*
import scair.dialects.affine.AffineMap
import scair.dialects.affine.AffineSet
import scair.ir.*
import scair.scairdl.constraints.BaseAttr
import scair.scairdl.constraints.ConstraintContext

import scala.util.Failure
import scala.util.Success
import scala.util.Try

// ██████╗░ ██╗░░░██╗ ██╗ ██╗░░░░░ ████████╗ ██╗ ███╗░░██╗
// ██╔══██╗ ██║░░░██║ ██║ ██║░░░░░ ╚══██╔══╝ ██║ ████╗░██║
// ██████╦╝ ██║░░░██║ ██║ ██║░░░░░ ░░░██║░░░ ██║ ██╔██╗██║
// ██╔══██╗ ██║░░░██║ ██║ ██║░░░░░ ░░░██║░░░ ██║ ██║╚████║
// ██████╦╝ ╚██████╔╝ ██║ ███████╗ ░░░██║░░░ ██║ ██║░╚███║
// ╚═════╝░ ░╚═════╝░ ╚═╝ ╚══════╝ ░░░╚═╝░░░ ╚═╝ ╚═╝░░╚══╝

// ██████╗░ ██╗ ░█████╗░ ██╗░░░░░ ███████╗ ░█████╗░ ████████╗
// ██╔══██╗ ██║ ██╔══██╗ ██║░░░░░ ██╔════╝ ██╔══██╗ ╚══██╔══╝
// ██║░░██║ ██║ ███████║ ██║░░░░░ █████╗░░ ██║░░╚═╝ ░░░██║░░░
// ██║░░██║ ██║ ██╔══██║ ██║░░░░░ ██╔══╝░░ ██║░░██╗ ░░░██║░░░
// ██████╔╝ ██║ ██║░░██║ ███████╗ ███████╗ ╚█████╔╝ ░░░██║░░░
// ╚═════╝░ ╚═╝ ╚═╝░░╚═╝ ╚══════╝ ╚══════╝ ░╚════╝░ ░░░╚═╝░░░

val I1 = IntegerType(IntData(1), Signless)
val I32 = IntegerType(IntData(32), Signless)
val I64 = IntegerType(IntData(64), Signless)

/*≡==--==≡≡≡≡==--=≡≡*\
||    SIGNEDNESS   ||
\*≡==---==≡≡==---==≡*/

sealed abstract class Signedness(override val name: String, val dat: String)
    extends DataAttribute[String](name, dat) {
  override def custom_print(p: Printer) = p.print(dat)
}

case object Signed extends Signedness("signed", "si")
case object Unsigned extends Signedness("unsigned", "ui")
case object Signless extends Signedness("signless", "i")

/*≡==--==≡≡≡≡==--=≡≡*\
||    FLOAT TYPE    ||
\*≡==---==≡≡==---==≡*/

abstract class FloatType(override val name: String)
    extends ParametrizedAttribute {
  override def parameters: Seq[Attribute | Seq[Attribute]] = Seq()
}

case class Float16Type() extends FloatType("builtin.f16") with TypeAttribute {
  override def custom_print(p: Printer) = p.print("f16")
}

case class Float32Type() extends FloatType("builtin.f32") with TypeAttribute {
  override def custom_print(p: Printer) = p.print("f32")
}

case class Float64Type() extends FloatType("builtin.f64") with TypeAttribute {
  override def custom_print(p: Printer) = p.print("f64")
}

case class Float80Type() extends FloatType("builtin.f80") with TypeAttribute {
  override def custom_print(p: Printer) = p.print("f80")
}

case class Float128Type() extends FloatType("builtin.f128") with TypeAttribute {
  override def custom_print(p: Printer) = p.print("f128")
}

/*≡==--==≡≡≡≡==--=≡≡*\
||     INT DATA     ||
\*≡==---==≡≡==---==≡*/

case class IntData(val value: Long)
    extends DataAttribute[Long]("builtin.int_attr", value)
    derives TransparentData {
  override def custom_print(p: Printer) = p.print(value.toString)
}

/*≡==--==≡≡≡≡==--=≡≡*\
||  INTEGER TYPE    ||
\*≡==---==≡≡==---==≡*/

case class IntegerType(val width: IntData, val sign: Signedness)
    extends ParametrizedAttribute
    with TypeAttribute {

  override def name: String = "builtin.int_type"
  override def parameters: Seq[Attribute | Seq[Attribute]] = Seq(width, sign)

  override def custom_print(p: Printer) =
    p.print(sign)
    p.print(width)

}

/*≡==--==≡≡≡≡==--=≡≡*\
|| INTEGER ATTRIBUTE ||
\*≡==---==≡≡==---==≡*/

case class IntegerAttr(
    val value: IntData,
    val typ: IntegerType | IndexType
) extends ParametrizedAttribute {

  def this(value: IntData) = this(value, I64)

  override def name: String = "builtin.integer_attr"
  override def parameters: Seq[Attribute | Seq[Attribute]] = Seq(value, typ)

  override def custom_print(p: Printer) = (value, typ) match {
    case (IntData(1), IntegerType(IntData(1), Signless)) => p.print("true")
    case (IntData(0), IntegerType(IntData(1), Signless)) => p.print("false")
    case (_, IntegerType(IntData(64), Signless))         => p.print(value)
    case (_, _) => p.print(value, " : ", typ)(using 0)
  }

}

/*≡==--==≡≡≡≡==--=≡≡*\
||    FLOAT DATA    ||
\*≡==---==≡≡==---==≡*/

case class FloatData(val value: Double)
    extends DataAttribute[Double]("builtin.float_data", value)
    derives TransparentData {
  override def custom_print(p: Printer) = p.print(value.toString)
}

/*≡==--==≡≡≡≡==--=≡≡*\
||  FLOAT ATTRIBUTE ||
\*≡==---==≡≡==---==≡*/

case class FloatAttr(val value: FloatData, val typ: FloatType)
    extends ParametrizedAttribute {

  override def name: String = "builtin.float_attr"
  override def parameters: Seq[Attribute | Seq[Attribute]] = Seq(value, typ)

  override def custom_print(p: Printer) = typ match
    case Float64Type => p.print(value)
    case _           => p.print(value, " : ", typ)(using 0)

}

/*≡==--==≡≡≡≡==--=≡≡*\
||   INDEX TYPE     ||
\*≡==---==≡≡==---==≡*/

case class IndexType() extends ParametrizedAttribute with TypeAttribute {
  override def name: String = "builtin.index"
  override def custom_print(p: Printer) = p.print("index")
  override def parameters: Seq[Attribute | Seq[Attribute]] = Seq()
}

/*≡==--==≡≡≡≡==--=≡≡*\
|| ARRAY ATTRIBUTE  ||
\*≡==---==≡≡==---==≡*/

case class ArrayAttribute[D <: Attribute](val attrValues: Seq[D])
    extends DataAttribute[Seq[D]]("builtin.array_attr", attrValues) {

  override def custom_print(p: Printer) =
    p.printList(attrValues, "[", ", ", "]")

}

/*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
|| DICTIONARY ATTRIBUTE  ||
\*≡==---==≡≡≡≡≡≡≡==---==≡*/

case class DictionaryAttr(val entries: Map[String, Attribute])
    extends DataAttribute[Map[String, Attribute]](
      "builtin.dict_attr",
      entries
    ) {

  override def custom_print(p: Printer) =
    p.printAttrDict(entries)

}

/*≡==--==≡≡≡≡==--=≡≡*\
|| STRING ATTRIBUTE ||
\*≡==---==≡≡==---==≡*/
case class StringData(val stringLiteral: String)
    extends DataAttribute("builtin.string", stringLiteral)
    derives TransparentData {

  override def custom_print(p: Printer) =
    p.print("\"", stringLiteral, "\"")(using 0)

}

/*≡==--==≡≡≡≡==--=≡≡*\
||   SHAPED TYPE    ||
\*≡==---==≡≡==---==≡*/

trait ShapedType extends TypeAttribute {
  def getNumDims: Int
  def getShape: Seq[Long]
  def elementCount: Long = getShape.product
}

/*≡==--==≡≡≡≡==--=≡≡*\
||   TENSOR TYPE    ||
\*≡==---==≡≡==---==≡*/

abstract class TensorType extends ParametrizedAttribute, TypeAttribute {
  def elementType: Attribute
}

case class RankedTensorType(
    override val elementType: Attribute,
    val shape: ArrayAttribute[IntData],
    val encoding: Option[Attribute] = None
) extends TensorType,
      ShapedType {

  override def name: String = "builtin.ranked_tensor"

  override def parameters: Seq[Attribute | Seq[Attribute]] =
    shape +: elementType +: encoding.toSeq

  override def getNumDims = shape.attrValues.length
  override def getShape = shape.attrValues.map(_.data)

  override def custom_print(p: Printer) =
    p.print("tensor<")
    shape.attrValues.foreach(s =>
      s match
        case IntData(-1) => p.print("?")
        case d           => p.print(d)
      p.print("x")
    )
    p.print(elementType)
    if encoding.isDefined then p.print(", ", encoding)(using indentLevel = 0)
    p.print(">")

}

case class UnrankedTensorType(override val elementType: Attribute)
    extends TensorType {
  override def name: String = "builtin.unranked_tensor"

  override def parameters: Seq[Attribute | Seq[Attribute]] =
    Seq(elementType)

  override def custom_print(p: Printer) =
    p.print("tensor<*x", elementType, ">")(using indentLevel = 0)

}

/*≡==--==≡≡≡≡==--=≡≡*\
||   MEMREF TYPE    ||
\*≡==---==≡≡==---==≡*/

abstract class MemrefType extends ParametrizedAttribute, TypeAttribute {
  def elementType: Attribute
}

case class RankedMemrefType(
    override val elementType: Attribute,
    val shape: ArrayAttribute[IntData],
    val encoding: Option[Attribute] = None
) extends MemrefType,
      ShapedType {

  override def name: String = "builtin.ranked_memref"

  override def parameters: Seq[Attribute | Seq[Attribute]] =
    shape +: elementType +: encoding.toSeq

  override def getNumDims = shape.attrValues.length
  override def getShape = shape.attrValues.map(_.data)

  override def custom_print(p: Printer) =
    p.print("memref<")
    shape.attrValues.foreach(s =>
      s match
        case IntData(-1) => p.print("?")
        case d           => p.print(d)
      p.print("x")
    )

    p.print(elementType, ">")(using indentLevel = 0)

}

case class UnrankedMemrefType(override val elementType: Attribute)
    extends MemrefType {

  override def name: String = "builtin.unranked_memref"

  override def parameters: Seq[Attribute | Seq[Attribute]] =
    Seq(elementType)

  override def custom_print(p: Printer) =
    p.print("tensor<*x", elementType, ">")(using indentLevel = 0)

}

/*≡==--==≡≡≡≡==--=≡≡*\
||   VECTOR TYPE    ||
\*≡==---==≡≡==---==≡*/

case class VectorType(
    val elementType: Attribute,
    val shape: ArrayAttribute[IntData],
    val scalableDims: ArrayAttribute[IntData]
) extends ParametrizedAttribute,
      ShapedType {

  override def name: String = "builtin.vector_type"

  override def parameters: Seq[Attribute | Seq[Attribute]] =
    Seq(shape, elementType, scalableDims)

  override def getNumDims = shape.attrValues.length
  override def getShape = shape.attrValues.map(_.data)

  override def custom_print(p: Printer): Unit =

    p.print("vector<")
    p.printListF(
      shape zip scalableDims,
      (size, scalable) =>
        if scalable.data != 0 then
          p.print("[", size, "]")(using indentLevel = 0)
        else p.print(size),
      sep = "x"
    )
    p.print("x", elementType, ">")(using indentLevel = 0)

}

/*≡==--==≡≡≡≡==--=≡≡*\
|| SYMBOL REF ATTR  ||
\*≡==---==≡≡==---==≡*/

case class SymbolRefAttr(
    val rootRef: StringData,
    val nestedRefs: Seq[StringData] = Seq()
) extends ParametrizedAttribute {

  override def name: String = "builtin.symbol_ref"

  override def parameters: Seq[Attribute | Seq[Attribute]] =
    Seq(rootRef, nestedRefs)

  override def custom_print(p: Printer) =
    p.printListF(
      rootRef +: nestedRefs,
      ref => p.print("@", ref.data)(using indentLevel = 0),
      sep = "::"
    )
    (rootRef +: nestedRefs).map(_.data).map("@" + _).mkString("::")

}

/*≡==--==≡≡≡≡==--=≡≡*\
|| DenseArrayAttr   ||
\*≡==---==≡≡==---==≡*/

case class DenseArrayAttr(
    val typ: IntegerType | FloatType,
    val data: Seq[IntegerAttr] | Seq[FloatAttr]
) extends ParametrizedAttribute
    with Seq[Attribute] {

  override def name: String = "builtin.dense_array"
  override def parameters: Seq[Attribute | Seq[Attribute]] = Seq(typ, data)

  override def custom_verify(): Either[String, Unit] =
    if !data.forall(_ match {
        case IntegerAttr(_, eltyp) => eltyp == typ
        case FloatAttr(_, eltyp)   => eltyp == typ
      })
    then Left("Element types do not match the dense array type")
    else Right(())

  override def custom_print(p: Printer) =
    p.print("array<", typ)(using indentLevel = 0)
    if data.nonEmpty then p.print(": ")
    p.printListF(
      data,
      {
        case IntegerAttr(value, _) => p.print(value)
        case FloatAttr(value, _)   => p.print(value)
      }
    )
    p.print(">")

  // Seq methods
  def apply(idx: Int): Attribute = data.apply(idx)

  def length: Int = data.length

  def iterator: Iterator[Attribute] = data.iterator
}

/*≡==--==≡≡≡≡==--=≡≡*\
||  FunctionType    ||
\*≡==---==≡≡==---==≡*/

case class FunctionType(
    val inputs: Seq[Attribute],
    val outputs: Seq[Attribute]
) extends ParametrizedAttribute
    with TypeAttribute {

  override def name: String = "builtin.function_type"

  override def parameters: Seq[Attribute | Seq[Attribute]] =
    Seq(inputs, outputs)

  override def custom_print(p: Printer) =
    p.print("(")
    p.printList(inputs)
    p.print(") -> ")
    outputs match
      case Seq(single) => p.print(single)
      case s           => p.printList(s, "(", ", ", ")")

}

/*≡==--==≡≡≡≡==--=≡≡*\
|| DenseIntOrFPAttr ||
\*≡==---==≡≡==---==≡*/

type TensorLiteralArray =
  ArrayAttribute[IntegerAttr] | ArrayAttribute[FloatAttr]

case class DenseIntOrFPElementsAttr(
    val typ: TensorType | MemrefType | VectorType,
    val data: TensorLiteralArray
) extends ParametrizedAttribute {

  override def name: String = "builtin.dense"
  override def parameters: Seq[Attribute | Seq[Attribute]] = Seq(typ, data)

  def elementType = typ match {
    case x: TensorType => x.elementType
    case x: MemrefType => x.elementType
    case x: VectorType => x.elementType
  }

  val int_or_float = BaseAttr[IntegerType | FloatType]()

  override def custom_verify(): Either[String, Unit] =
    Try(int_or_float.verify(elementType, new ConstraintContext())) match {
      case Success(_) =>
        Try(
          for (x <- data.attrValues)
            int_or_float.verify(x, new ConstraintContext())
        ) match {
          case Success(_) => Right(())
          case Failure(e) => Left(e.getMessage)
        }
      case Failure(e) => Left(e.getMessage)
    }

  override def custom_print(p: Printer) =
    val values = data.attrValues(0) match {
      case x: IntegerAttr =>
        for (a <- data.attrValues) yield a.asInstanceOf[IntegerAttr].value
      case y: FloatAttr =>
        for (a <- data.attrValues) yield a.asInstanceOf[FloatAttr].value
    }
    p.print("dense<")
    values match
      case Seq(single) => p.print(single)
      case s           => p.printList(s, "[", ", ", "]")
    p.print("> : ")
    p.print(typ)

}

/*≡==--==≡≡≡≡==--=≡≡*\
||  AFFINE MAP ATTR ||
\*≡==---==≡≡==---==≡*/

case class AffineMapAttr(val affine_map: AffineMap)
    extends DataAttribute[AffineMap]("builtin.affine_map", affine_map)
    derives TransparentData {

  override def custom_print(p: Printer) =
    p.print("affine_map<", affine_map.toString, ">")(using indentLevel = 0)

}

/*≡==--==≡≡≡≡==--=≡≡*\
||  AFFINE SET ATTR ||
\*≡==---==≡≡==---==≡*/
// note: in mlir terms this is called an IntegerSetAttr

case class AffineSetAttr(val affine_set: AffineSet)
    extends DataAttribute[AffineSet]("builtin.affine_set", affine_set)
    derives TransparentData {

  override def custom_print(p: Printer) =
    p.print("affine_set<", affine_set.toString, ">")(using indentLevel = 0)

}

/*≡==--==≡≡≡≡==--=≡≡*\
||   OPERATIONS    ||
\*≡==---==≡≡==---==≡*/

// ==------== //
//  ModuleOp  //
// ==------== //

object ModuleOp extends OperationCompanion {
  override def name = "builtin.module"

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      parser: Parser
  ): P[Operation] =
    P(
      parser.Region()
    ).map((x: Region) => ModuleOp(regions = Seq(x)))
  // ==----------------------== //

}

case class ModuleOp(
    override val operands: Seq[Value[Attribute]] = Seq(),
    override val successors: Seq[Block] = Seq(),
    override val results: Seq[Result[Attribute]] = Seq(),
    override val regions: Seq[Region] = Seq(),
    override val properties: Map[String, Attribute] =
      Map.empty[String, Attribute],
    override val attributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends BaseOperation(
      name = "builtin.module",
      operands,
      successors,
      results,
      regions,
      properties,
      attributes
    ) {

  override def custom_print(
      p: Printer
  )(using indentLevel: Int) =
    p.print("builtin.module ", regions(0))

}

object UnrealizedConversionCastOp extends OperationCompanion {
  override def name = "builtin.unrealized_conversion_cast"
}

case class UnrealizedConversionCastOp(
    override val operands: Seq[Value[Attribute]] = Seq(),
    override val successors: Seq[Block] = Seq(),
    override val results: Seq[Result[Attribute]] = Seq(),
    override val regions: Seq[Region] = Seq(),
    override val properties: Map[String, Attribute] =
      Map.empty[String, Attribute],
    override val attributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends BaseOperation(
      name = "builtin.unrealized_conversion_cast",
      operands,
      successors,
      results,
      regions,
      properties,
      attributes
    )

val BuiltinDialect = Dialect(Seq(ModuleOp, UnrealizedConversionCastOp), Seq())

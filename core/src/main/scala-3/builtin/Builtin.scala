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
  override def custom_print = dat
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
  override def custom_print = "f16"
}

case class Float32Type() extends FloatType("builtin.f32") with TypeAttribute {
  override def custom_print = "f32"
}

case class Float64Type() extends FloatType("builtin.f64") with TypeAttribute {
  override def custom_print = "f64"
}

case class Float80Type() extends FloatType("builtin.f80") with TypeAttribute {
  override def custom_print = "f80"
}

case class Float128Type() extends FloatType("builtin.f128") with TypeAttribute {
  override def custom_print = "f128"
}

/*≡==--==≡≡≡≡==--=≡≡*\
||     INT DATA     ||
\*≡==---==≡≡==---==≡*/

case class IntData(val value: Long)
    extends DataAttribute[Long]("builtin.int_attr", value)
    derives TransparentData {
  override def custom_print = value.toString
}

/*≡==--==≡≡≡≡==--=≡≡*\
||  INTEGER TYPE    ||
\*≡==---==≡≡==---==≡*/

case class IntegerType(val width: IntData, val sign: Signedness)
    extends ParametrizedAttribute
    with TypeAttribute {

  override def name: String = "builtin.int_type"
  override def parameters: Seq[Attribute | Seq[Attribute]] = Seq(width, sign)

  override def custom_print = sign match {
    case Signless => s"${sign.custom_print}${width.custom_print}"
    case Signed   => s"${sign.custom_print}${width.custom_print}"
    case Unsigned => s"${sign.custom_print}${width.custom_print}"
  }

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

  override def custom_print = (value, typ) match {
    case (IntData(1), IntegerType(IntData(1), Signless)) => "true"
    case (IntData(0), IntegerType(IntData(1), Signless)) => "false"
    case (_, IntegerType(IntData(64), Signless)) => s"${value.custom_print}"
    case (_, _) => s"${value.custom_print} : ${typ.custom_print}"
  }

}

/*≡==--==≡≡≡≡==--=≡≡*\
||    FLOAT DATA    ||
\*≡==---==≡≡==---==≡*/

case class FloatData(val value: Double)
    extends DataAttribute[Double]("builtin.float_data", value)
    derives TransparentData {
  override def custom_print = value.toString
}

/*≡==--==≡≡≡≡==--=≡≡*\
||  FLOAT ATTRIBUTE ||
\*≡==---==≡≡==---==≡*/

case class FloatAttr(val value: FloatData, val typ: FloatType)
    extends ParametrizedAttribute {

  override def name: String = "builtin.float_attr"
  override def parameters: Seq[Attribute | Seq[Attribute]] = Seq(value, typ)

  override def custom_print = (value, typ) match {
    case (_, Float64Type) => s"${value.custom_print}"
    case (_, _)           => s"${value.custom_print} : ${typ.custom_print}"
  }

}

/*≡==--==≡≡≡≡==--=≡≡*\
||   INDEX TYPE     ||
\*≡==---==≡≡==---==≡*/

case class IndexType() extends ParametrizedAttribute with TypeAttribute {
  override def name: String = "builtin.index"
  override def custom_print = "index"
  override def parameters: Seq[Attribute | Seq[Attribute]] = Seq()
}

/*≡==--==≡≡≡≡==--=≡≡*\
|| ARRAY ATTRIBUTE  ||
\*≡==---==≡≡==---==≡*/

case class ArrayAttribute[D <: Attribute](val attrValues: Seq[D])
    extends DataAttribute[Seq[D]]("builtin.array_attr", attrValues) {

  override def custom_print =
    "[" + attrValues.map(x => x.custom_print).mkString(", ") + "]"

}

/*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
|| DICTIONARY ATTRIBUTE  ||
\*≡==---==≡≡≡≡≡≡≡==---==≡*/

case class DictionaryAttr(val entries: Map[String, Attribute])
    extends DataAttribute[Map[String, Attribute]](
      "builtin.dict_attr",
      entries
    ) {

  override def custom_print =
    "{" + entries.map((name, attr) => s"$name = $attr").mkString(", ") + "}"

}

/*≡==--==≡≡≡≡==--=≡≡*\
|| STRING ATTRIBUTE ||
\*≡==---==≡≡==---==≡*/
case class StringData(val stringLiteral: String)
    extends DataAttribute("builtin.string", stringLiteral)
    derives TransparentData {
  override def custom_print = "\"" + stringLiteral + "\""
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

  override def custom_print: String = {
    val shapeString =
      (shape.data.map(x =>
        if (x.data == -1) "?" else x.custom_print
      ) :+ elementType.custom_print)
        .mkString("x")

    val encodingString = encoding match {
      case Some(x) => x.custom_print
      case None    => ""
    }

    return s"tensor<${shapeString}${encodingString}>"
  }

}

case class UnrankedTensorType(override val elementType: Attribute)
    extends TensorType {
  override def name: String = "builtin.unranked_tensor"

  override def parameters: Seq[Attribute | Seq[Attribute]] =
    Seq(elementType)

  override def custom_print = s"tensor<*x${elementType.custom_print}>"
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

  override def custom_print: String = {

    val shapeString =
      (shape.map(x =>
        if (x.data == -1) "?" else x.custom_print
      ) :+ elementType.custom_print)
        .mkString("x")

    return s"memref<${shapeString}>"
  }

}

case class UnrankedMemrefType(override val elementType: Attribute)
    extends MemrefType {

  override def name: String = "builtin.unranked_memref"

  override def parameters: Seq[Attribute | Seq[Attribute]] =
    Seq(elementType)

  override def custom_print = s"tensor<*x${elementType.custom_print}>"
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

  override def custom_print: String = {

    val shapeString =
      ((shape, scalableDims).zipped
        .map((size, scalable) =>
          if scalable.data != 0 then s"[${size.data}]" else s"${size.data}"
        ) :+ elementType.custom_print)
        .mkString("x")

    return s"vector<${shapeString}>"
  }

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

  override def custom_print =
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

  override def custom_print = {

    return s"array<${typ.custom_print}${if data.isEmpty then "" else ": "}${data
        .map(_ match {
          case IntegerAttr(value, _) => value.custom_print
          case FloatAttr(value, _)   => value.custom_print
        })
        .mkString(", ")}>"
  }

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

  override def custom_print = {
    val inputsString = inputs.map(_.custom_print).mkString(", ")
    val outputsString = outputs.map(_.custom_print).mkString(", ")
    outputs.length match {
      case 1 => s"(${inputsString}) -> ${outputsString}"
      case _ => s"(${inputsString}) -> (${outputsString})"
    }
  }

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

  override def custom_print = {

    val values = data.attrValues(0) match {
      case x: IntegerAttr =>
        for (a <- data.attrValues) yield a.asInstanceOf[IntegerAttr].value
      case y: FloatAttr =>
        for (a <- data.attrValues) yield a.asInstanceOf[FloatAttr].value
    }

    return s"dense<${
        if (values.size == 1) { values(0).custom_print }
        else { values.map(_.custom_print).mkString("[", ", ", "]") }
      }> : ${typ.custom_print}"
  }

}

/*≡==--==≡≡≡≡==--=≡≡*\
||  AFFINE MAP ATTR ||
\*≡==---==≡≡==---==≡*/

case class AffineMapAttr(val affine_map: AffineMap)
    extends DataAttribute[AffineMap]("builtin.affine_map", affine_map)
    derives TransparentData {

  override def custom_print = s"affine_map<${affine_map}>"
}

/*≡==--==≡≡≡≡==--=≡≡*\
||  AFFINE SET ATTR ||
\*≡==---==≡≡==---==≡*/
// note: in mlir terms this is called an IntegerSetAttr

case class AffineSetAttr(val affine_set: AffineSet)
    extends DataAttribute[AffineSet]("builtin.affine_set", affine_set)
    derives TransparentData {

  override def custom_print = s"affine_set<${affine_set}>"
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
      parser.Region
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

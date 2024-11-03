package scair.dialects.builtin

import scala.compiletime.ops.string
import scala.collection.{immutable, mutable}

import scair.dialects.affine.{AffineMap, AffineSet}
import scair.scairdl.constraints.{BaseAttr, EqualAttr, ConstraintContext}
import scair.{Parser, Printer}
import scair.ir._

import scair.Parser.whitespace
import fastparse._
import scair.exceptions.VerifyException

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

def I1 = IntegerType(IntData(1), Signless)
def I32 = IntegerType(IntData(32), Signless)
def I64 = IntegerType(IntData(64), Signless)

////////////////
// SIGNEDNESS //
////////////////

sealed abstract class Signedness(override val name: String, val dat: String)
    extends DataAttribute[String](name, dat) {
  override def custom_print = dat
}
case object Signed extends Signedness("signed", "si")
case object Unsigned extends Signedness("unsigned", "ui")
case object Signless extends Signedness("signless", "i")

////////////////
// FLOAT TYPE //
////////////////

abstract class FloatType(val namee: String) extends ParametrizedAttribute(namee)

case object Float16Type extends FloatType("builtin.f16") with TypeAttribute {
  override def custom_print = "f16"
}
case object Float32Type extends FloatType("builtin.f32") with TypeAttribute {
  override def custom_print = "f32"
}
case object Float64Type extends FloatType("builtin.f64") with TypeAttribute {
  override def custom_print = "f64"
}
case object Float80Type extends FloatType("builtin.f80") with TypeAttribute {
  override def custom_print = "f80"
}
case object Float128Type extends FloatType("builtin.f128") with TypeAttribute {
  override def custom_print = "f128"
}

//////////////
// INT DATA //
//////////////

case class IntData(val value: Long)
    extends DataAttribute[Long]("builtin.int_attr", value) {
  override def custom_print = value.toString
}

//////////////////
// INTEGER TYPE //
//////////////////

case class IntegerType(val width: IntData, val sign: Signedness)
    extends ParametrizedAttribute("builtin.int_type", Seq(width, sign))
    with TypeAttribute {
  override def custom_print = sign match {
    case Signless => s"${sign.custom_print}${width.custom_print}"
    case Signed   => s"${sign.custom_print}${width.custom_print}"
    case Unsigned => s"${sign.custom_print}${width.custom_print}"
  }
}

///////////////////////
// INTEGER ATTRIBUTE //
///////////////////////

case class IntegerAttr(
    val value: IntData,
    val typ: IntegerType | IndexType.type
) extends ParametrizedAttribute("builtin.integer_attr", Seq(value, typ)) {

  def this(value: IntData) = this(value, I64)

  override def custom_print = (value, typ) match {
    case (IntData(1), IntegerType(IntData(1), Signless)) => "true"
    case (IntData(0), IntegerType(IntData(1), Signless)) => "false"
    case (_, IntegerType(IntData(64), Signless)) => s"${value.custom_print}"
    case (_, _) => s"${value.custom_print} : ${typ.custom_print}"
  }
}

////////////////
// FLOAT DATA //
////////////////

case class FloatData(val value: Double)
    extends DataAttribute[Double]("builtin.float_data", value) {
  override def custom_print = value.toString
}

/////////////////////
// FLOAT ATTRIBUTE //
/////////////////////

case class FloatAttr(val value: FloatData, val typ: FloatType)
    extends ParametrizedAttribute("builtin.float_attr", Seq(value, typ)) {
  override def custom_print = (value, typ) match {
    case (_, Float64Type) => s"${value.custom_print}"
    case (_, _)           => s"${value.custom_print} : ${typ.custom_print}"
  }
}

////////////////
// INDEX TYPE //
////////////////

case object IndexType
    extends ParametrizedAttribute("builtin.index")
    with TypeAttribute {
  override def custom_print = "index"
}

/////////////////////
// ARRAY ATTRIBUTE //
/////////////////////

case class ArrayAttribute[D <: Attribute](val attrValues: Seq[D])
    extends DataAttribute[Seq[D]]("builtin.array_attr", attrValues) {
  override def custom_print =
    "[" + attrValues.map(x => x.custom_print).mkString(", ") + "]"
}

//////////////////////
// STRING ATTRIBUTE //
//////////////////////

// shortened definition, does not include type information
case class StringData(val stringLiteral: String)
    extends DataAttribute("builtin.string", stringLiteral) {
  override def custom_print = "\"" + stringLiteral + "\""
}

/////////////////
// SHAPED TYPE //
/////////////////

trait ShapedType extends TypeAttribute

/////////////////
// TENSOR TYPE //
/////////////////

abstract class TensorType(
    override val name: String,
    val typ: Attribute,
    val features: Seq[Attribute]
) extends ParametrizedAttribute(name, features)
    with TypeAttribute

case class RankedTensorType(
    val dimensionList: ArrayAttribute[IntData],
    override val typ: Attribute,
    val encoding: Option[Attribute]
) extends TensorType(
      name = "builtin.ranked_tensor",
      typ,
      features = dimensionList +:
        typ +:
        encoding.toSeq
    ) {

  override def custom_print: String = {

    val shapeString =
      (dimensionList.data.map(x =>
        if (x.data == -1) "?" else x.custom_print
      ) :+ typ.custom_print)
        .mkString("x")

    val encodingString = encoding match {
      case Some(x) => x.custom_print
      case None    => ""
    }

    return s"tensor<${shapeString}${encodingString}>"
  }
}

case class UnrankedTensorType(override val typ: Attribute)
    extends TensorType("builtin.unranked_tensor", typ, Seq(typ)) {
  override def custom_print = s"tensor<*x${typ.custom_print}>"
}

//////////////////////////
// SYMBOL REF ATTRIBUTE //
//////////////////////////

case class SymbolRefAttr(
    val rootRef: StringData,
    val nestedRefs: ArrayAttribute[StringData]
) extends ParametrizedAttribute(
      name = "builtin.symbol_ref",
      Seq(rootRef, nestedRefs)
    ) {
  override def custom_print =
    s"@${rootRef.data}::${nestedRefs.data.map(x => s"@${x.data}").mkString("::")}"
}

////////////////////
// DenseArrayAttr //
////////////////////

case class DenseArrayAttr(
    val typ: Attribute,
    val data: Seq[Attribute]
) extends ParametrizedAttribute("builtin.dense", typ +: data) {

  override def custom_verify(): Unit =
    typ match {
      case _: IntegerType | _: FloatType => ()
      case _ =>
        throw new VerifyException(
          "Element type is not an integer or float type"
        )
    }
    if !data.forall(_ match {
        case IntegerAttr(_, eltyp) => eltyp == typ
        case FloatAttr(_, eltyp)   => eltyp == typ
        case _ =>
          throw new VerifyException("Element type is not an integer or float")
      })
    then
      throw new VerifyException(
        "Element types do not match the dense array type"
      )

  override def custom_print = {

    return s"array<${typ.custom_print}${if data.isEmpty then "" else ": "}${data
        .map(_ match {
          case IntegerAttr(value, _) => value.custom_print
          case FloatAttr(value, _)   => value.custom_print
        })
        .mkString(", ")}>"
  }
}

//////////////////////////////
// DenseIntOrFPElementsAttr //
//////////////////////////////
//
// TO-DO : it can also parse a vector type or a memref type

type TensorLiteralArray =
  ArrayAttribute[IntegerAttr] | ArrayAttribute[FloatAttr]

case class DenseIntOrFPElementsAttr(
    val typ: TensorType,
    val data: TensorLiteralArray
) extends ParametrizedAttribute("builtin.dense") {

  val int_or_float = BaseAttr[IntegerType | FloatType]()

  override def custom_verify(): Unit =
    int_or_float.verify(typ.typ, new ConstraintContext())
    for (x <- data.attrValues) int_or_float.verify(x, new ConstraintContext())

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

/////////////////////
// AFFINE MAP ATTR //
/////////////////////

case class AffineMapAttr(val affine_map: AffineMap)
    extends DataAttribute[AffineMap]("builtin.affine_map", affine_map) {

  override def custom_print = s"affine_map<${affine_map}>"
}

/////////////////////
// AFFINE SET ATTR //
/////////////////////
// note: in mlir terms this is called an IntegerSetAttr

case class AffineSetAttr(val affine_set: AffineSet)
    extends DataAttribute[AffineSet]("builtin.affine_set", affine_set) {

  override def custom_print = s"affine_set<${affine_set}>"
}

////////////////
// OPERATIONS //
////////////////

// ==------== //
//  ModuleOp  //
// ==------== //

object ModuleOp {
  // ==--- Custom Parsing ---== //
  def parse[$: P](parser: Parser): P[Operation] = P(
    "builtin.module" ~ parser.Region.rep(exactly = 1)
  ).map((x: Seq[Region]) => ModuleOp(regions = x.to(ListType)))
  // ==----------------------== //
}

case class ModuleOp(
    override val operands: ListType[Value[Attribute]] = ListType(),
    override val successors: ListType[Block] = ListType(),
    override val results: ListType[Value[Attribute]] = ListType(),
    override val regions: ListType[Region] = ListType(),
    override val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends RegisteredOperation(name = "builtin.module") {
  override def custom_print(
      p: Printer
  ): String =
    s"builtin.module ${p.printRegion(regions(0))}"
}

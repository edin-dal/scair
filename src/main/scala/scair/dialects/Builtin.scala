package scair.dialects.builtin

import scala.compiletime.ops.string
import scala.collection.{immutable, mutable}

import scair.dialects.affine.{AffineMap, AffineSet}
import scair.dialects.irdl.{BaseAttr, EqualAttr, ConstraintContext}
import scair.{
  ListType,
  DictType,
  Attribute,
  TypeAttribute,
  ParametrizedAttribute,
  DataAttribute,
  Value,
  Block,
  Region,
  Operation,
  RegisteredOperation,
  Parser,
  Printer
}

import scair.Parser.whitespace
import fastparse._

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
  override def toString = dat
}
case object Signed extends Signedness("signed", "si")
case object Unsigned extends Signedness("unsigned", "ui")
case object Signless extends Signedness("signless", "i")

////////////////
// FLOAT TYPE //
////////////////

abstract class FloatType(val namee: String) extends ParametrizedAttribute(namee)

case object Float16Type extends FloatType("builtin.f16") with TypeAttribute {
  override def toString = "f16"
}
case object Float32Type extends FloatType("builtin.f32") with TypeAttribute {
  override def toString = "f32"
}
case object Float64Type extends FloatType("builtin.f64") with TypeAttribute {
  override def toString = "f64"
}
case object Float80Type extends FloatType("builtin.f80") with TypeAttribute {
  override def toString = "f80"
}
case object Float128Type extends FloatType("builtin.f128") with TypeAttribute {
  override def toString = "f128"
}

//////////////
// INT DATA //
//////////////

case class IntData(val value: Long)
    extends DataAttribute[Long]("builtin.int_attr", value) {
  override def toString = value.toString
}

//////////////////
// INTEGER TYPE //
//////////////////

case class IntegerType(val width: IntData, val sign: Signedness)
    extends ParametrizedAttribute("builtin.int_type", Seq(width, sign))
    with TypeAttribute {
  override def toString = sign match {
    case Signless => s"$sign$width"
    case Signed   => s"$sign$width"
    case Unsigned => s"$sign$width"
  }
}

///////////////////////
// INTEGER ATTRIBUTE //
///////////////////////

case class IntegerAttr(val value: IntData, val typ: IntegerType)
    extends ParametrizedAttribute("builtin.integer_attr", Seq(value, typ)) {

  def this(value: IntData) = this(value, I64)

  override def toString = (value, typ) match {
    case (IntData(1), IntegerType(IntData(1), Signless)) => "true"
    case (IntData(0), IntegerType(IntData(1), Signless)) => "false"
    case (_, IntegerType(IntData(64), Signless))         => s"${value}"
    case (_, _)                                          => s"${value} : ${typ}"
  }
}

////////////////
// FLOAT DATA //
////////////////

case class FloatData(val value: Double)
    extends DataAttribute[Double]("builtin.float_data", value) {
  override def toString = value.toString
}

/////////////////////
// FLOAT ATTRIBUTE //
/////////////////////

case class FloatAttr(val value: FloatData, val typ: FloatType)
    extends ParametrizedAttribute("builtin.float_attr", Seq(value, typ)) {
  override def toString = (value, typ) match {
    case (_, Float64Type) => s"${value}"
    case (_, _)           => s"${value} : ${typ}"
  }
}

////////////////
// INDEX TYPE //
////////////////

case object IndexType
    extends ParametrizedAttribute("builtin.index")
    with TypeAttribute {
  override def toString = "index"
}

/////////////////////
// ARRAY ATTRIBUTE //
/////////////////////

case class ArrayAttribute[D <: Attribute](val attrValues: Seq[D])
    extends DataAttribute[Seq[D]]("builtin.array_attr", attrValues) {
  override def toString =
    "[" + attrValues.map(x => x.toString).mkString(", ") + "]"
}

//////////////////////
// STRING ATTRIBUTE //
//////////////////////

// shortened definition, does not include type information
case class StringData(val stringLiteral: String)
    extends DataAttribute("builtin.string", stringLiteral) {
  override def toString = "\"" + stringLiteral + "\""
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
    with ShapedType

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

  override def toString: String = {

    val shapeString =
      (dimensionList.data.map(x =>
        if (x.data == -1) "?" else x.toString
      ) :+ typ.toString)
        .mkString("x")

    val encodingString = encoding match {
      case Some(x) => x.toString
      case None    => ""
    }

    return s"tensor<${shapeString}${encodingString}>"
  }
}

case class UnrankedTensorType(override val typ: Attribute)
    extends TensorType("builtin.unranked_tensor", typ, Seq(typ)) {
  override def toString = s"tensor<*x${typ.toString}>"
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
  override def toString =
    s"@${rootRef.data}::${nestedRefs.data.map(x => s"@${x.data}").mkString("::")}"
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

  val type_cnstr = BaseAttr[IntegerType | FloatType]()
  val TLA_cnstr = EqualAttr(typ.typ)

  override def custom_verify(): Unit =
    type_cnstr.verify(typ.typ, new ConstraintContext())
    for (x <- data.attrValues) TLA_cnstr.verify(x, new ConstraintContext())

  override def toString() = {

    val values = data.attrValues(0) match {
      case x: IntegerAttr =>
        for (a <- data.attrValues) yield a.asInstanceOf[IntegerAttr].value
      case y: FloatAttr =>
        for (a <- data.attrValues) yield a.asInstanceOf[FloatAttr].value
    }

    return s"dense<${
        if (values.size == 1) { values(0) }
        else { values.mkString("[", ", ", "]") }
      }> : ${typ}"
  }
}

/////////////////////
// AFFINE MAP ATTR //
/////////////////////

case class AffineMapAttr(val affine_map: AffineMap)
    extends DataAttribute[AffineMap]("builtin.affine_map", affine_map) {

  override def toString = s"affine_map<${affine_map}>"
}

/////////////////////
// AFFINE SET ATTR //
/////////////////////
// note: in mlir terms this is called an IntegerSetAttr

case class AffineSetAttr(val affine_set: AffineSet)
    extends DataAttribute[AffineSet]("builtin.affine_set", affine_set) {

  override def toString = s"affine_set<${affine_set}>"
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

package scair.dialects.builtin

import scair.{Attribute, TypeAttribute, ParametrizedAttribute, DataAttribute}
import scala.compiletime.ops.string

def I1 = IntegerType(1, Signless)
def I32 = IntegerType(32, Signless)
def I64 = IntegerType(64, Signless)

////////////////
// SIGNEDNESS //
////////////////

sealed trait Signedness
case object Signed extends Signedness
case object Unsigned extends Signedness
case object Signless extends Signedness

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

//////////////////
// INTEGER TYPE //
//////////////////

case class IntegerType(val width: Int, val sign: Signedness)
    extends ParametrizedAttribute("builtin.int_type")
    with TypeAttribute {
  override def toString = sign match {
    case Signless => s"i$width"
    case Signed   => s"si$width"
    case Unsigned => s"ui$width"
  }
}

case class IntAttr(val value: Long)
    extends DataAttribute[Long]("builtin.int_attr", value)

///////////////////////
// INTEGER ATTRIBUTE //
///////////////////////

case class IntegerAttr(val value: Long, val typ: IntegerType)
    extends ParametrizedAttribute("builtin.integer_attr") {

  def this(value: Long) = this(value, I64)
  override def toString = (value, typ) match {
    case (1, IntegerType(1, Signless))  => "true"
    case (0, IntegerType(1, Signless))  => "false"
    case (_, IntegerType(64, Signless)) => s"${value}"
    case (_, _)                         => s"${value} : ${typ}"
  }
}

/////////////////////
// FLOAT ATTRIBUTE //
/////////////////////

case class FloatAttr(val value: Double, val typ: FloatType)
    extends ParametrizedAttribute("builtin.float_attr") {
  override def toString = (value, typ) match {
    case (_, Float64Type) => s"${value}"
    case (_, _)           => s"${value} : ${typ}"
  }
}

//////////////////
// INTEGER TYPE //
//////////////////

case class IntegerAttr(val value: Int, val typ: IntegerType)
    extends ParametrizedAttribute("builtin.integer_attr") {
  override def toString = (value, typ) match {
    case (1, IntegerType(1, Signless))  => "true"
    case (0, IntegerType(1, Signless))  => "false"
    case (_, IntegerType(64, Signless)) => s"${value}"
    case (_, _)                         => s"${value} : ${typ}"
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
case class StringAttribute(val stringLiteral: String)
    extends DataAttribute("builtin.string_attribute", stringLiteral) {
  override def toString = "\"" + stringLiteral + "\""
}

/////////////////
// TENSOR TYPE //
/////////////////

case class RankedTensorType(
    val dimensionList: ArrayAttribute[IntAttr],
    val typ: Attribute,
    val encoding: Option[Attribute]
) extends ParametrizedAttribute(
      name = "builtin.ranked_tensor",
      dimensionList +:
        typ +:
        encoding.toSeq
    )
    with TypeAttribute {

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

case class UnrankedTensorType(val typ: Attribute)
    extends ParametrizedAttribute("builtin.unranked_tensor", Seq(typ))
    with TypeAttribute {
  override def toString = s"tensor<*x${typ.toString}>"
}

//////////////////////////
// SYMBOL REF ATTRIBUTE //
//////////////////////////

case class SymbolRefAttr(
    val rootRef: StringAttribute,
    val nestedRefs: ArrayAttribute[StringAttribute]
) extends ParametrizedAttribute(
      name = "builtin.symbol_ref",
      Seq(rootRef, nestedRefs)
    ) {
  override def toString =
    s"@${rootRef.data}::${nestedRefs.data.map(x => s"@${x.data}").mkString("::")}"
}

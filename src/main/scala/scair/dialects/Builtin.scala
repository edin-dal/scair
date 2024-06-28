package scair.dialects.builtin

import scair.Attribute
import scair.TypeAttribute

sealed trait Signedness
case object Signed extends Signedness
case object Unsigned extends Signedness
case object Signless extends Signedness

case object Float16Type extends TypeAttribute("builtin.f16") {
  override def toString = "f16"
}
case object Float32Type extends TypeAttribute("builtin.f32") {
  override def toString = "f32"
}
case object Float64Type extends TypeAttribute("builtin.f64") {
  override def toString = "f64"
}
case object Float80Type extends TypeAttribute("builtin.f80") {
  override def toString = "f80"
}
case object Float128Type extends TypeAttribute("builtin.f128") {
  override def toString = "f128"
}

case class IntegerType(val width: Int, val sign: Signedness)
    extends TypeAttribute("builtin.int_type") {
  override def toString = sign match {
    case Signless => s"i$width"
    case Signed   => s"si$width"
    case Unsigned => s"ui$width"
  }
}

case object IndexType extends TypeAttribute("builtin.index") {
  override def toString = "index"
}

case class ArrayAttribute(val attrValues: Seq[Attribute])
    extends Attribute("builtin.array_attribute") {
  override def toString =
    "[" + attrValues.map(x => x.toString).mkString(", ") + "]"
}

// shortened definition, does not include type information
case class StringAttribute(val stringLiteral: String)
    extends Attribute("builtin.string_attribute") {
  override def toString = "\"" + stringLiteral + "\""
}

case class RankedTensorType(
    val dimensionList: Seq[Int],
    val typ: Attribute,
    val encoding: Option[Attribute]
) extends TypeAttribute("builtin.ranked_tensor") {

  override def toString: String = {

    val shapeString =
      (dimensionList.map(x => if (x == -1) "?" else x.toString) :+ typ.toString)
        .mkString("x")

    val encodingString = encoding match {
      case Some(x) => x.toString
      case None    => ""
    }

    return s"tensor<${shapeString}${encodingString}>"
  }
}

case class UnrankedTensorType(val typ: Attribute)
    extends TypeAttribute("builtin.unranked_tensor") {
  override def toString = s"tensor<*x${typ.toString}>"
}

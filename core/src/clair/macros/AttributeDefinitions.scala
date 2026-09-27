package scair.clair.macros

import scair.ir.Attribute

import scala.quoted.*

case class AttributeParamDef(
    val name: String,
    val tpe: Type[? <: Attribute],
) {}

case class AttributeDef(
    val name: String,
    val attributes: Seq[AttributeParamDef] = Seq(),
):

  def allDefsWithIndex =
    attributes.zipWithIndex

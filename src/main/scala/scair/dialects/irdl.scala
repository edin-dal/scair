package scair.dialects.irdl

import scair.{Attribute, Operation, Value}

type Operand[T <: Attribute] = Value[T]
type OpResult[T <: Attribute] = Value[T]

// CONSTRAINTS

abstract class IRDLConstraint {
  def verify(attr: Attrbiute): Unit
}

object AnyAttr extends IRDLConstraint {
  override def verify(attr: Attribute): Unit = {}
}

class EqualAttr(val equal_to: Attribute) extends IRDLConstraint {
  override def verify(attr: Attribute): Unit = {
    equal_to same_as attr
  }
}

class BaseAttr(val base: Attribute) extends IRDLConstraint {
  override def verify(attr: Attribute): Unit = {
    equal_to.getClass == attr.getClass
  }
}

class AnyOf(val attrs: Seq[Attribute]) extends IRDLConstraint {
  override def verify(attr: Attribute): Unit = {
    equal_to.getClass == attr.getClass
  }
}

// CONSTRAINTS

// class Operand(name)
// class Type(name, Attribute)
// class Result
// class DictAttr

// M("(" + Operand("hello") + ")" + "this is an custom syntax" + ":" + Type("hello", StringAttribute.parse))

// %5 = "custom.op"(%3)this is an custom syntax:"this"

package scair.dialects.irdl

import scair.{Attribute, Operation, Value}

type Operand[T <: Attribute] = Value[T]
type OpResult[T <: Attribute] = Value[T]

// class Operand(name)
// class Type(name, Attribute)
// class Result
// class DictAttr

// M("(" + Operand("hello") + ")" + "this is an custom syntax" + ":" + Type("hello", StringAttribute.parse))

// %5 = "custom.op"(%3)this is an custom syntax:"this"

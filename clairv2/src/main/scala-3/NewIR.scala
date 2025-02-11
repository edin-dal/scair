package test.newir

import scair.ir.{Value, Attribute, ListType, DictType}
import scair.dialects.builtin.IntegerAttr
import test.codegen.*
import test.showmacros.typeToString

import scala.reflect.ClassTag
import scala.compiletime.*
import scala.deriving.*
import scala.reflect.*

/*≡≡=---=≡≡≡≡≡≡≡≡≡=---=≡≡*\
||   DIFFERENT CLASSES   ||
\*≡==----=≡≡≡≡≡≡≡=----==≡*/

// case class Operand[+T <: Attribute](
//     typp: T
// ) extends Value[T](typp)

// case class Result[+T <: Attribute](
//     typp: T
// ) extends Value[T](typp)

object ValueConversions {

  given opToVal[T <: Attribute]: Conversion[Operand[T], Value[T]] with
    def apply(op: Operand[T]): Value[T] = op.value

  given resToVal[T <: Attribute]: Conversion[Result[T], Value[T]] with
    def apply(op: Result[T]): Value[T] = op.value

  given valToOp[T <: Attribute]: Conversion[Value[T], Operand[T]] with
    def apply(value: Value[T]): Operand[T] = Operand(value)

  given valToRes[T <: Attribute]: Conversion[Value[T], Result[T]] with
    def apply(value: Value[T]): Result[T] = Result(value)

}

// TODO change Operand to an Alias of Value
case class Operand[+T <: Attribute](val value: Value[T])
case class Result[+T <: Attribute](val value: Value[T])

case class Successor()
case class Region()

case class Property[+T <: Attribute](
    val typ: T
)

case class Attr[+T <: Attribute](
    val typ: T
)

type Variadic[T] = Seq[T]

abstract class Operation()

trait ADTCompanion {
  val getMLIRRealm: MLIRRealm[_]
}

class ADTOperation() extends Operation

abstract class MLIROp(
    val name: String,
    val operands: ListType[Value[Attribute]] = ListType(),
    val successors: ListType[Successor] = ListType(),
    val results_types: ListType[Attribute] = ListType(),
    val regions: ListType[Region] = ListType(),
    val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends Operation {
  val results: ListType[Value[Attribute]] = results_types.map(Value(_))
}

class RegisteredOp[T <: ADTOperation: ClassTag](
    name: String,
    operands: ListType[Value[Attribute]] = ListType(),
    successors: ListType[Successor] = ListType(),
    results_types: ListType[Attribute] = ListType(),
    regions: ListType[Region] = ListType(),
    dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends MLIROp(
      name,
      operands,
      successors,
      results_types,
      regions,
      dictionaryProperties,
      dictionaryAttributes
    )

class UnregisteredOp(
    name: String,
    operands: ListType[Value[Attribute]] = ListType(),
    successors: ListType[Successor] = ListType(),
    results_types: ListType[Attribute] = ListType(),
    regions: ListType[Region] = ListType(),
    dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends MLIROp(
      name,
      operands,
      successors,
      results_types,
      regions,
      dictionaryProperties,
      dictionaryAttributes
    )

object MLIRRealm {}

trait MLIRRealm[T <: ADTOperation]() {

  def unverify(op: T): RegisteredOp[T]

  def verify(op: RegisteredOp[_]): T

}

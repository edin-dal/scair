package scair.ir

import scala.collection.mutable.LinkedHashMap
import scala.collection.mutable.ListBuffer

// ██╗ ██████╗░
// ██║ ██╔══██╗
// ██║ ██████╔╝
// ██║ ██╔══██╗
// ██║ ██║░░██║
// ╚═╝ ╚═╝░░╚═╝

// ██╗░░░██╗ ████████╗ ██╗ ██╗░░░░░ ░██████╗
// ██║░░░██║ ╚══██╔══╝ ██║ ██║░░░░░ ██╔════╝
// ██║░░░██║ ░░░██║░░░ ██║ ██║░░░░░ ╚█████╗░
// ██║░░░██║ ░░░██║░░░ ██║ ██║░░░░░ ░╚═══██╗
// ╚██████╔╝ ░░░██║░░░ ██║ ███████╗ ██████╔╝
// ░╚═════╝░ ░░░╚═╝░░░ ╚═╝ ╚══════╝ ╚═════╝░

/*≡==--==≡≡≡≡≡≡≡==--=≡≡*\
||      OP INPUTS      ||
\*≡==---==≡≡≡≡≡==---==≡*/
// for ClairV2

type Variadic[T] = Seq[T]

type Operand[+T <: Attribute] = Value[T]
case class Result[+T <: Attribute](val value: Value[T])

type Successor = Block

case class Property[+T <: Attribute](
    val typ: T
)

case class Attr[+T <: Attribute](
    val typ: T
)

/*≡==--==≡≡≡≡≡==--=≡≡*\
||    CONVERSIONS    ||
\*≡==---==≡≡≡==---==≡*/

object ValueConversions {

  given resToVal[T <: Attribute]: Conversion[Result[T], Value[T]] with
    def apply(op: Result[T]): Value[T] = op.value

  given valToRes[T <: Attribute]: Conversion[Value[T], Result[T]] with
    def apply(value: Value[T]): Result[T] = Result(value)

}

/*≡==--==≡≡≡==--=≡≡*\
||      UTILS      ||
\*≡==---==≡==---==≡*/

val DictType = LinkedHashMap
type DictType[A, B] = LinkedHashMap[A, B]

val ListType = ListBuffer
type ListType[A] = ListBuffer[A]

extension (dt: DictType[String, Attribute]) {

  def checkandget(
      key: String,
      op_name: String,
      expected_type: String
  ): Attribute = {
    dt.get(key) match {
      case Some(b) => b
      case None =>
        throw new Exception(
          s"Operation '${op_name}' must include an attribute named '${key}' of type '${}'"
        )
    }
  }

}

extension (lt: ListType[Value[Attribute]]) {

  def updateOperandsAndUses(use: Use, newValue: Value[Attribute]): Unit = {
    newValue.uses += use
    lt.update(use.index, newValue)
  }

}

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

type Operand[+T <: Attribute] = Value[T]
case class Result[+T <: Attribute](val value: Value[T])

case class Property[+T <: Attribute](
    val typ: T
)

case class Attr[+T <: Attribute](
    val typ: T
)

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

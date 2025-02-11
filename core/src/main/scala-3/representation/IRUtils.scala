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

/*≡==--==≡≡≡≡≡≡==--=≡≡*\
||      OP INPUT      ||
\*≡==---==≡≡≡≡==---==≡*/

// type Operand[+T <: Attribute] = Value[T]
// type Result[+T <: Attribute] = Value[T]

// case class Property[+T <: Attribute]() {
//   var value: T = ???
// }

// case class Attr[+T <: Attribute]() {
//   var value: T = ???
// }

// sealed abstract class Operation(
//     val name: String,
//     val operands: ListType[Value[Attribute]] = ListType(),
//     val successors: ListType[Block] = ListType(),
//     results_types: ListType[Attribute] = ListType(),
//     val regions: ListType[Region] = ListType(),
//     val dictionaryProperties: DictType[String, Attribute] =
//       DictType.empty[String, Attribute],
//     val dictionaryAttributes: DictType[String, Attribute] =
//       DictType.empty[String, Attribute]
// ) extends OpTrait {

// case class Operand[T <: Attribute | AttributeFE]() extends Input[T]
// case class Result[T <: Attribute | AttributeFE]() extends Input[T]
// case class Region() extends Input[Nothing]
// case class Successor() extends Input[Nothing]
// case class Property[T <: Attribute | AttributeFE]() extends Input[T]
// case class Attr[T <: Attribute | AttributeFE]() extends Input[T]

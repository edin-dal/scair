package scair.ir

import scala.annotation.tailrec
import scala.collection.mutable.LinkedHashMap
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.Map

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

/*≡==--==≡≡≡≡≡==--=≡≡*\
||      IR NODE      ||
\*≡==---==≡≡≡==---==≡*/

trait IRNode:
  def parent: Option[IRNode]

  @tailrec
  final def topLevel: IRNode =
    parent match
      case Some(p) => p.topLevel
      case None    => this

  final def isAncestor(other: IRNode): Boolean =
    other.parent match
      case Some(parent) if parent == this => true
      case Some(parent)                   => isAncestor(parent)
      case None                           => false
      case null                           => false

  def deepCopy(using
      blockMapper: Map[Block, Block] = Map.empty,
      valueMapper: Map[Value[Attribute], Value[Attribute]] = Map.empty,
  ): IRNode

  def recomputeOpOrder(): Unit = ???

/*≡==--==≡≡≡≡≡≡≡==--=≡≡*\
||      OP INPUTS      ||
\*≡==---==≡≡≡≡≡==---==≡*/
// for ClairV2

type Successor = Block

/*≡==--==≡≡≡==--=≡≡*\
||      UTILS      ||
\*≡==---==≡==---==≡*/

val ListType = ListBuffer
type ListType[A] = ListBuffer[A]

val DictType = LinkedHashMap
type DictType[A, B] = LinkedHashMap[A, B]

extension (dt: DictType[String, Attribute])

  def checkandget(
      key: String,
      opName: String,
      expected_type: String,
  ): Attribute =
    dt.get(key) match
      case Some(b) => b
      case None    =>
        throw new Exception(
          s"Operation '$opName' must include an attribute named '$key' of type '${}'"
        )

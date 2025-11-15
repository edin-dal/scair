package scair.utils

import scala.annotation.switch
import scala.annotation.tailrec
import scala.collection.*

//
// ██╗ ███╗░░██╗ ████████╗ ██████╗░ ██╗░░░██╗ ░██████╗ ██╗ ██╗░░░██╗ ███████╗
// ██║ ████╗░██║ ╚══██╔══╝ ██╔══██╗ ██║░░░██║ ██╔════╝ ██║ ██║░░░██║ ██╔════╝
// ██║ ██╔██╗██║ ░░░██║░░░ ██████╔╝ ██║░░░██║ ╚█████╗░ ██║ ╚██╗░██╔╝ █████╗░░
// ██║ ██║╚████║ ░░░██║░░░ ██╔══██╗ ██║░░░██║ ░╚═══██╗ ██║ ░╚████╔╝░ ██╔══╝░░
// ██║ ██║░╚███║ ░░░██║░░░ ██║░░██║ ╚██████╔╝ ██████╔╝ ██║ ░░╚██╔╝░░ ███████╗
// ╚═╝ ╚═╝░░╚══╝ ░░░╚═╝░░░ ╚═╝░░╚═╝ ░╚═════╝░ ╚═════╝░ ╚═╝ ░░░╚═╝░░░ ╚══════╝
//
// ██╗░░░░░ ██╗ ░██████╗ ████████╗
// ██║░░░░░ ██║ ██╔════╝ ╚══██╔══╝
// ██║░░░░░ ██║ ╚█████╗░ ░░░██║░░░
// ██║░░░░░ ██║ ░╚═══██╗ ░░░██║░░░
// ███████╗ ██║ ██████╔╝ ░░░██║░░░
// ╚══════╝ ╚═╝ ╚═════╝░ ░░░╚═╝░░░
//

trait IntrusiveNode[A]:
  this: A =>

  private[utils] final var _prev: Option[A] = None
  private[utils] final var _next: Option[A] = None

  inline final def prev: Option[A] = _prev
  inline final def next: Option[A] = _next

  inline final private[utils] def prev_=(p: Option[A]) =
    _prev = p

  inline final private[utils] def next_=(n: Option[A]) =
    _next = n

class IntrusiveList[A <: IntrusiveNode[A]] extends mutable.Buffer[A]:

  private final var _head: Option[A] = None
  private final var _last: Option[A] = None

  override def iterator: Iterator[A] = new AbstractIterator[A]:
    private var current = _head
    def hasNext = current.isDefined
    def next(): A =
      val res = current.get
      current = res.next
      res

  @tailrec
  private def applyRec(idx: Int, current: A): A =
    idx: @switch match
      case 0 => current
      case _ =>
        current.next match
          case Some(n) => applyRec(idx - 1, n)
          case None    => throw new IndexOutOfBoundsException("Out of bounds")

  override def apply(i: Int): A =
    _head match
      case Some(h) => applyRec(i, h)
      case None    => throw new IndexOutOfBoundsException("Empty")

  override def clear(): Unit =
    _head = None
    _last = None

  override def remove(idx: Int): A =
    val v = apply(idx)
    subtractOne(v)
    v

  override def subtractOne(x: A): this.type =
    x.prev match
      case Some(p) => p.next = x.next
      case None    => _head = x.next
    x.next match
      case Some(n) => n.prev = x.prev
      case None    => _last = x.prev
    this

  def update(c: A, e: A): Unit =
    // TODO: how should we deal with that
    if c.eq(e) then return

    if _head.isDefined && _head.get.eq(c) then _head = Some(e)
    if _last.isDefined && _last.get.eq(c) then _last = Some(e)

    val p = c.prev
    val n = c.next

    if p.isDefined then p.get.next = Some(e)
    if n.isDefined then n.get.prev = Some(e)

    e.prev = p
    e.next = n

    c.prev = None
    c.next = None

  override def update(idx: Int, elem: A): Unit =
    update(apply(idx), elem)

  @tailrec
  private def lengthRec(current: Option[A], acc: Int): Int =
    current match
      case Some(n) => lengthRec(n.next, acc + 1)
      case None    => acc

  override def length: Int =
    lengthRec(_head, 0)

  @tailrec
  private final def foreachRec[U](current: Option[A])(f: A => U): Unit =
    current match
      case Some(n) =>
        f(n)
        foreachRec(n.next)(f)
      case None => ()

  // Override of foreach to accomodate side effects on the list chaining.
  override def foreach[U](f: A => U): Unit =
    foreachRec(_head)(f)

  override def addOne(elem: A): this.type =
    _last match
      case None =>
        _head = Some(elem)
        _last = _head
      case Some(last) =>
        last.next = Some(elem)
        elem.prev = Some(last)
        _last = Some(elem)
    this

  override def prepend(elem: A): this.type =
    _head match
      case None =>
        _head = Some(elem)
        _last = _head
      case Some(head) =>
        head.prev = Some(elem)
        elem.next = Some(head)
        _head = Some(elem)
    this

  override def insert(idx: Int, elem: A): Unit =
    if idx == length then addOne(elem)
    else insert(apply(idx), elem)

  def insert(c: A, e: A): Unit =
    if _head.isDefined && _head.get.eq(c) then prepend(e)
    else
      val prev = c.prev.get
      prev.next = Some(e)
      e.prev = Some(prev)
      e.next = Some(c)
      c.prev = Some(e)

  override def insertAll(idx: Int, elems: IterableOnce[A]): Unit =
    if idx == length then addAll(elems)
    else insertAll(apply(idx), elems)

  def insertAll(at: A, elems: IterableOnce[A]): Unit =
    elems.iterator.foreach(e => insert(at, e))

  override def patchInPlace(
      from: Int,
      patch: IterableOnce[A],
      replaced: Int
  ): this.type =
    var i = from
    patch.iterator
      .take(replaced)
      .foreach(p =>
        update(i, p)
        i += 1
      )
    this

  override def remove(idx: Int, count: Int): Unit =
    (1 to count).foreach(_ => remove(idx))

object IntrusiveList:

  def apply[A <: IntrusiveNode[A]](elems: A*): IntrusiveList[A] =
    from(elems)

  def empty[A <: IntrusiveNode[A]]: IntrusiveList[A] = new IntrusiveList[A]

  def from[A <: IntrusiveNode[A]](i: IterableOnce[A]) =
    val list = new IntrusiveList[A]
    list.addAll(i)

  def unapplySeq[A <: IntrusiveNode[A]](list: IntrusiveList[A]): Some[Seq[A]] =
    Some(list.toSeq)

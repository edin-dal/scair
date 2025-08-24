package scair.utils

import scala.annotation.switch
import scala.annotation.tailrec
import scala.collection.*

trait IntrusiveNode[A]:
  this: A =>

  private[utils] final var _prev: Option[IntrusiveNode[A]] = None
  private[utils] final var _next: Option[IntrusiveNode[A]] = None

  final def prev: Option[IntrusiveNode[A]] = _prev
  final def next: Option[IntrusiveNode[A]] = _next

  inline private[utils] def prev_=(p: Option[IntrusiveNode[A]]) =
    _prev = p

  inline private[utils] def next_=(n: Option[IntrusiveNode[A]]) =
    _next = n

class IntrusiveList[A] extends mutable.Buffer[A]:

  private var _head: Option[IntrusiveNode[A]] = None
  private var _last: Option[IntrusiveNode[A]] = None

  override def iterator: Iterator[A] = new AbstractIterator[A]:
    private var current = _head
    def hasNext = current.isDefined
    def next(): A =
      val res = current.get
      current = res.next
      res.asInstanceOf[A]

  private def applyRec(idx: Int, current: IntrusiveNode[A]): A =
    idx: @switch match
      case 0 => current.asInstanceOf[A]
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
    val v = x.asInstanceOf[IntrusiveNode[A]]
    v.prev match
      case Some(p) => p.next = v.next
      case None    => _head = v.next
    v.next match
      case Some(n) => n.prev = v.prev
      case None    => _last = v.prev
    v.asInstanceOf[A]
    this

  override def update(idx: Int, elem: A): Unit =
    val e = elem.asInstanceOf[IntrusiveNode[A]]
    val current = apply(idx).asInstanceOf[IntrusiveNode[A]]

    // TODO: how should we deal with that
    if current.eq(e) then return

    if _head.isDefined && _head.get.eq(current) then _head = Some(e)
    if _last.isDefined && _last.get.eq(current) then _last = Some(e)

    val p = current.prev.asInstanceOf[Option[IntrusiveNode[A]]]
    val n = current.next.asInstanceOf[Option[IntrusiveNode[A]]]

    if p.isDefined then p.get.next = Some(e)
    if n.isDefined then n.get.prev = Some(e)

    e.prev = p
    e.next = n

    current.prev = None
    current.next = None

  @tailrec
  private def lengthRec(current: Option[IntrusiveNode[A]], acc: Int): Int =
    current match
      case Some(n) => lengthRec(n.next, acc + 1)
      case None    => acc

  override def length: Int =
    lengthRec(_head, 0)

  override def addOne(elem: A): this.type =
    _last match
      case None =>
        _head = Some(elem.asInstanceOf[IntrusiveNode[A]])
        _last = _head
      case Some(last) =>
        last.next = Some(elem.asInstanceOf[IntrusiveNode[A]])
        elem.asInstanceOf[IntrusiveNode[A]].prev = Some(last)
        _last = Some(elem.asInstanceOf[IntrusiveNode[A]])
    this

  override def prepend(elem: A): this.type =
    _head match
      case None =>
        _head = Some(elem.asInstanceOf[IntrusiveNode[A]])
        _last = _head
      case Some(head) =>
        head.prev = Some(elem.asInstanceOf[IntrusiveNode[A]])
        elem.asInstanceOf[IntrusiveNode[A]].next = Some(head)
        _head = Some(elem.asInstanceOf[IntrusiveNode[A]])
    this

  override def insert(idx: Int, elem: A): Unit =
    if idx == length then addOne(elem)
    else if idx == 0 then prepend(elem)
    else
      val current = apply(idx)
      val prev = current.asInstanceOf[IntrusiveNode[A]].prev
      elem.asInstanceOf[IntrusiveNode[A]].next = Some(
        current.asInstanceOf[IntrusiveNode[A]]
      )
      current.asInstanceOf[IntrusiveNode[A]].prev = Some(
        elem.asInstanceOf[IntrusiveNode[A]]
      )
      elem.asInstanceOf[IntrusiveNode[A]].prev = prev
      prev match
        case Some(p) => p.next = Some(elem.asInstanceOf[IntrusiveNode[A]])
        case None    => _head = Some(elem.asInstanceOf[IntrusiveNode[A]])

  def insert(c: A, elem: A): Unit =
    if _last.isDefined && _last.get.eq(c.asInstanceOf[IntrusiveNode[A]]) then
      addOne(elem)
    else if _head.isDefined && _head.get.eq(c.asInstanceOf[IntrusiveNode[A]])
    then prepend(elem)
    else
      val current = c.asInstanceOf[IntrusiveNode[A]]
      val next = current.next
      elem.asInstanceOf[IntrusiveNode[A]].prev = Some(current)
      current.next = Some(elem.asInstanceOf[IntrusiveNode[A]])
      elem.asInstanceOf[IntrusiveNode[A]].next = next
      next match
        case Some(n) => n.prev = Some(elem.asInstanceOf[IntrusiveNode[A]])
        case None    => _last = Some(elem.asInstanceOf[IntrusiveNode[A]])

  override def insertAll(idx: Int, elems: IterableOnce[A]): Unit =
    var i = 0
    elems.foreach(e =>
      insert(idx + i, e)
      i += 1
    )

  def insertAll(at: A, elems: IterableOnce[A]): Unit =
    var a = at
    elems.foreach(e =>
      insert(a, e)
      a = e
    )

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

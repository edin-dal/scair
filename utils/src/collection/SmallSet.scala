package scair.collection

import scala.annotation.tailrec
import scala.collection.mutable

/** A mutable set optimised for a handful of elements.
  *
  * Elements are kept in a small array and looked up linearly, avoiding the hash
  * table and per-element nodes of a HashSet. Past `threshold` elements, the set
  * switches to a HashSet so that large sets keep constant-time operations.
  *
  * @param initialCapacity
  *   The capacity of the initial array.
  * @param threshold
  *   The size past which the set switches to a HashSet.
  */
final class SmallSet[A](initialCapacity: Int = 2, threshold: Int = 8)
    extends mutable.AbstractSet[A]:

  private var small: Array[AnyRef] = new Array[AnyRef](initialCapacity)
  private var count: Int = 0
  // Non-null once the set has outgrown the array; small is then dropped.
  private var large: mutable.HashSet[A] = null

  private inline def elem(i: Int): A = small(i).asInstanceOf[A]

  @tailrec
  private def indexOf(e: A, i: Int = 0): Int =
    if i >= count then -1
    else if elem(i) == e then i
    else indexOf(e, i + 1)

  override def size: Int = if large ne null then large.size else count
  override def knownSize: Int = size
  override def isEmpty: Boolean = size == 0

  def contains(e: A): Boolean =
    if large ne null then large.contains(e) else indexOf(e) >= 0

  def addOne(e: A): this.type =
    if large ne null then large.addOne(e)
    else if indexOf(e) < 0 then
      if count == threshold then
        large = new mutable.HashSet[A]
        var i = 0
        while i < count do
          large.addOne(elem(i))
          i += 1
        small = null
        count = 0
        large.addOne(e)
      else
        if count == small.length then
          small = java.util.Arrays.copyOf(small, count * 2)
        small(count) = e.asInstanceOf[AnyRef]
        count += 1
    this

  def subtractOne(e: A): this.type =
    if large ne null then large.subtractOne(e)
    else
      val i = indexOf(e)
      if i >= 0 then
        count -= 1
        // Order is irrelevant; fill the hole with the last element.
        small(i) = small(count)
        small(count) = null
    this

  def clear(): Unit =
    if large ne null then large.clear()
    else
      java.util.Arrays.fill(small, 0, count, null)
      count = 0

  override def filterInPlace(p: A => Boolean): this.type =
    if large ne null then large.filterInPlace(p)
    else
      var i = 0
      var kept = 0
      while i < count do
        val e = elem(i)
        if p(e) then
          small(kept) = e.asInstanceOf[AnyRef]
          kept += 1
        i += 1
      java.util.Arrays.fill(small, kept, count, null)
      count = kept
    this

  override def foreach[U](f: A => U): Unit =
    if large ne null then large.foreach(f)
    else
      var i = 0
      while i < count do
        f(elem(i))
        i += 1

  def iterator: Iterator[A] =
    if large ne null then large.iterator
    else
      val end = count
      new Iterator[A]:
        private var i = 0
        def hasNext = i < end
        def next() =
          val e = elem(i)
          i += 1
          e

package scair.helpers

import scala.collection.LinearSeq

extension [T](inline seq: Seq[T])

  /** foreach inlined at the call site: no closure, no iterator, whether the
    * sequence is linked or indexed.
    */
  inline def foreachInline(inline f: T => Unit): Unit =
    seq match
      case linear: LinearSeq[T] @unchecked =>
        var current = linear
        while current.nonEmpty do
          f(current.head)
          current = current.tail
      case indexed =>
        val length = indexed.length
        var i = 0
        while i < length do
          f(indexed(i))
          i += 1

extension [T](inline seq: Iterable[T])

  inline def foreachWithIndex(inline f: (T, Int) => Unit) =
    var i = 0
    seq match
      case s: Seq[T] @unchecked =>
        s.foreachInline(element =>
          f(element, i)
          i += 1
        )
      case other =>
        other.foreach(element =>
          f(element, i)
          i += 1
        )

  transparent inline def mapWithIndex[O](inline f: (T, Int) => O) =
    var i = 0
    seq.map(element =>
      val n = f(element, i)
      i += 1
      n
    )

  transparent inline def flatMapWithIndex[O](
      inline f: (T, Int) => IterableOnce[O]
  ) =
    var i = 0
    seq.flatMap(element =>
      val n = f(element, i)
      i += 1
      n
    )

package scair.helpers

extension [T](inline seq: Iterable[T])

  inline def foreachWithIndex(inline f: (T, Int) => Unit) =
    var i = 0
    seq.foreach(element =>
      f(element, i)
      i = i + 1
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

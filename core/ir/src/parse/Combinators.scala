package scair.parse

import fastparse.*
import fastparse.Implicits.Repeater

import scala.annotation.tailrec
import scala.collection.mutable
import scala.collection.mutable.Builder

/*≡==--==≡≡≡≡==--=≡≡*\
|| COMMON FUNCTIONS ||
\*≡==---==≡≡==---==≡*/

extension [T](inline p: P[T])

  /** Make the parser optional, parsing defaults if otherwise failing.
    *
    * @todo:
    *   Figure out dark implicit magic to figure out magically that the default
    *   default is "T()".
    *
    * @param default
    *   The default value to use if the parser fails.
    * @return
    *   An optional parser, defaulting to default.
    */
  inline def orElse[$: P](inline default: => T): P[T] = P(
    p | Pass(default)
  )

  /** Like fastparse's flatMapX but capturing exceptions as standard parse
    * errors.
    *
    * @note
    *   flatMapX because it often yields more natural error positions.
    *
    * @param f
    *   The function to apply to the parsed value.
    * @return
    *   A parser that applies f to the parsed value, catching exceptions and
    *   turning them into parse errors.
    */
  inline def flatMapTry[$: P, V](inline f: T => P[V]): P[V] = P(
    p.flatMapX(parsed =>
      try f(parsed)
      catch
        case e: Exception =>
          Console.err.print(
            "WARNING: Caught an exception in parsing; this is deprecated, use fastparse's Fail instead.\n"
          )
          Fail(e.getMessage())
    )
  )

  // Replacement for fastparse's .opaque, with a by-name message, so as not to build it in the happy case.
  // TODO: Should that be contributed to fastparse's .opauqe or does it have a reason not to be?
  inline def explain[$: P as ctx](inline msg: => String): P[T] =
    val oldIndex = ctx.index
    val startTerminals = ctx.terminalMsgs
    val res = p

    val res2 =
      if res.isSuccess then ctx.freshSuccess(ctx.successValue)
      else ctx.freshFailure(oldIndex)

    if ctx.verboseFailures then
      ctx.terminalMsgs = startTerminals
      ctx.reportTerminalMsg(oldIndex, () => msg)

    res2.asInstanceOf[P[T]]

  /** Like fastparse's mapX but capturing exceptions as standard parse errors.
    *
    * @note
    *   flatMapX because it often yields more nat ural error positions.
    *
    * @param f
    *   The function to apply to the parsed value.
    * @return
    *   A parser that applies f to the parsed value, catching exceptions and
    *   turning them into parse errors.
    */
  inline def mapTry[$: P, V](inline f: T => V): P[V] = P(
    p.flatMapX(parsed =>
      try Pass(f(parsed))
      catch
        case e: Exception =>
          Console.err.print(
            "WARNING: Caught an exception in parsing; this is deprecated, use fastparse's Fail instead.\n"
          )
          Fail(e.getMessage())
    )
  )

@tailrec
def flatRepRec[$: P, V, T](
    i: Seq[T],
    f: T => P[V],
    running: P[Builder[V, Seq[V]]],
    sep: => P[Unit] = null,
    error: Int => String,
)(using
    whitespace: Whitespace
): P[Builder[V, Seq[V]]] =
  i match
    case head +: tail =>
      flatRepRec(
        tail,
        f,
        running.flatMap(builder =>
          (sep ~ f(head).map(builder.addOne)).explain(error(builder.knownSize))
        ),
        sep,
        error,
      )
    case Nil => running

extension [T](inline i: Seq[T])

  inline def flatRep[$: P, V](
      inline f: T => P[V],
      inline sep: => P[Unit] = null,
      inline error: Int => String,
  )(using
      whitespace: Whitespace
  ): P[Seq[V]] =
    i match
      case head +: tail =>
        val builder = Seq.newBuilder[V]
        flatRepRec(
          tail,
          f,
          f(head).map(builder.addOne).explain(error(0)),
          sep,
          error,
        ).map(_.result())
      case Nil => Pass(Seq.empty[V])

// See uses; enables .rep to concatenate parsed sequences
// TODO: Expose as nicer helper, but could'nt get it just right for now
def concatRepeater[T] = new Repeater[Seq[T], Seq[T]]:
  type Acc = mutable.Buffer[T]
  def initial = mutable.Buffer.empty[T]
  def accumulate(t: Seq[T], acc: mutable.Buffer[T]) = acc ++= t
  def result(acc: mutable.Buffer[T]) = acc.toSeq

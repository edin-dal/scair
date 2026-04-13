package scair.utils

/** Either an error message or an expected value of type T
  * @tparam T
  *   The expected value type
  * @note
  *   Implemented in terms of Either right now; only defined for readability for
  *   now.
  */
opaque type OK[+T] = Err | T
final case class Err(msg: String)

object OK:
  def apply[T](t: T): OK[T] = t
  def apply(): OK[Unit] = OK(())

  def unapply[T](ok: OK[T]): Option[T] = ok match
    case err: Err => None
    case t: T     => Some(t)

  given error: [T] => Conversion[Err, OK[T]] = x => x
  given discard: [T] => Conversion[OK[T], OK[Unit]] = x => ()
  given merge: [T] => Conversion[OK[T] | Err, OK[T]] = x => x

extension [T](ok: OK[T])

  inline def map[U](inline f: T => U): OK[U] =
    ok match
      case e: Err => e
      case t: T   => OK(f(t))

  inline def flatMap[U](inline f: T => OK[U]): OK[U] =
    ok match
      case e: Err => e
      case t: T   => f(t)

  inline def fold[C](inline fa: Err => C, inline fb: T => C): C =
    ok match
      case e: Err => fa(e)
      case t: T   => fb(t)

  inline def get: T =
    ok match
      case e: Err => throw new NoSuchElementException()
      case t: T   => t

  inline def getError: Err =
    ok match
      case e: Err => e
      case t: T   => throw new NoSuchElementException()

  inline def isOK: Boolean =
    !isError

  inline def isError: Boolean =
    ok.isInstanceOf[Err]

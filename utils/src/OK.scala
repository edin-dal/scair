package scair.utils

/** Either an error message or an expected value of type T
  * @tparam T
  *   The expected value type
  * @note
  *   Implemented in terms of Either right now; only defined for readability for
  *   now.
  */
type OK[+T] = Either[String, T]

package scair.exceptions

import scala.io.AnsiColor.*

// TODO: What's the complete parameter list of Exception??
class VerifyException(val msg: String) extends Exception(msg)

/** Base class for all ScaIR errors.
  *
  * @param msg:
  *   String \=> Error message to be displayed.
  */
abstract class ScaIError(msg: String) {
  val errorTag: String
  val errorMessage: String = msg.split("\n").map(errorTag + _).mkString("\n")
}

/** Error encompassing a verification error in ScaIR.
  *
  * @param msg:
  *   String \=> Error message to be displayed.
  */
class VError(
    msg: String
) extends ScaIError(msg) {
  override val errorTag = s"[${YELLOW}verification${RESET}]    "
}

/** Error encompassing a transformation error in ScaIR.
  *
  * @param msg:
  *   String \=> Error message to be displayed.
  */
class TError(
    msg: String
) extends ScaIError(msg) {
  override val errorTag = s"[${CYAN}transform${RESET}]    "
}

/** Error encompassing a internal system error in ScaIR.
  *
  * @param msg:
  *   String \=> Error message to be displayed.
  */
class ISError(
    msg: String
) extends ScaIError(msg) {
  override val errorTag = s"[${RED}internal${RESET}]    "
}

sealed abstract class Verified[+A, +B <: ScaIError] {}

final case class Success[+A, +B <: ScaIError](value: A) extends Verified[A, B]

final case class Error[+A, +B <: ScaIError](error: B) extends Verified[A, B]

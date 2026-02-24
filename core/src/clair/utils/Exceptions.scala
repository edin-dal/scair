package scair.clair

import fastparse.Parsed

import scala.io.AnsiColor.*

// ░█████╗░ ██╗░░░░░ ░█████╗░ ██╗ ██████╗░
// ██╔══██╗ ██║░░░░░ ██╔══██╗ ██║ ██╔══██╗
// ██║░░╚═╝ ██║░░░░░ ███████║ ██║ ██████╔╝
// ██║░░██╗ ██║░░░░░ ██╔══██║ ██║ ██╔══██╗
// ╚█████╔╝ ███████╗ ██║░░██║ ██║ ██║░░██║
// ░╚════╝░ ╚══════╝ ╚═╝░░╚═╝ ╚═╝ ╚═╝░░╚═╝

// ███████╗ ██╗░░██╗ ░█████╗░ ███████╗ ██████╗░ ████████╗ ██╗ ░█████╗░ ███╗░░██╗ ░██████╗
// ██╔════╝ ╚██╗██╔╝ ██╔══██╗ ██╔════╝ ██╔══██╗ ╚══██╔══╝ ██║ ██╔══██╗ ████╗░██║ ██╔════╝
// █████╗░░ ░╚███╔╝░ ██║░░╚═╝ █████╗░░ ██████╔╝ ░░░██║░░░ ██║ ██║░░██║ ██╔██╗██║ ╚█████╗░
// ██╔══╝░░ ░██╔██╗░ ██║░░██╗ ██╔══╝░░ ██╔═══╝░ ░░░██║░░░ ██║ ██║░░██║ ██║╚████║ ░╚═══██╗
// ███████╗ ██╔╝╚██╗ ╚█████╔╝ ███████╗ ██║░░░░░ ░░░██║░░░ ██║ ╚█████╔╝ ██║░╚███║ ██████╔╝
// ╚══════╝ ╚═╝░░╚═╝ ░╚════╝░ ╚══════╝ ╚═╝░░░░░ ░░░╚═╝░░░ ╚═╝ ░╚════╝░ ╚═╝░░╚══╝ ╚═════╝░

private val clairErr = s"[${RED}claierror$RESET]   "

/** Custom exception type for Clair errors. Does not fill in stack trace for
  * cleaner error output.
  *
  * @param name
  *   The name/category of the error.
  * @param message
  *   The detailed error message.
  */
class ClairCustomException(name: String, message: String)
    extends Exception(
      "\n" + clairErr + name + "\n" + clairErr + " => " + message + "\n"
    ):
  override def fillInStackTrace(): Throwable = this

/** Methods for handling Clair exceptions and parse errors. */

/** Methods for handling Clair exceptions and parse errors. */
object ClairExceptionMethods:

  /** Throws a parse error with formatted output and exits.
    *
    * @param input
    *   The input being parsed.
    * @param failure
    *   The fastparse failure object.
    */
  def throwParseError(input: String, failure: Parsed.Failure) =
    val traced = failure.extra.trace
    val msg =
      s"Parse error at $input:${failure.extra.input
          .prettyIndex(failure.index)}:\n\n${failure.extra.trace().aggregateMsg}"
    Console.err.print(msg)
    sys.exit(0)

  /** Throws a custom Clair error with formatted output and exits. For other
    * throwables, re-raises them.
    *
    * @param throwable
    *   The throwable to handle.
    */
  def throwCustomClairError(throwable: Throwable): Unit =
    throwable match
      case e: ClairCustomException =>
        Console.err.print(throwable.getMessage())
        sys.exit(0)
      case _ => throw throwable

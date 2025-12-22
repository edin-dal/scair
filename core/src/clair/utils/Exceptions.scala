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

class ClairCustomException(name: String, message: String)
    extends Exception(
      "\n" + clairErr + name + "\n" + clairErr + " => " + message + "\n"
    ):
  override def fillInStackTrace(): Throwable = this

object ClairExceptionMethods:

  def throwParseError(input: String, failure: Parsed.Failure) =
    val traced = failure.extra.trace
    val msg =
      s"Parse error at $input:${failure.extra.input
          .prettyIndex(failure.index)}:\n\n${failure.extra.trace().aggregateMsg}"
    Console.err.print(msg)
    sys.exit(0)

  def throwCustomClairError(throwable: Throwable): Unit =
    throwable match
      case e: ClairCustomException =>
        Console.err.print(throwable.getMessage())
        sys.exit(0)
      case _ => throw throwable

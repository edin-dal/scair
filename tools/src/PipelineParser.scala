package scair.tools

import fastparse.*
import scair.MLContext
import scair.transformations.*

/** Parses a pipeline directly into registered pass instances. */
final class PipelineParser(context: MLContext):

  def parse(input: String): Either[String, Seq[ModulePass]] =
    fastparse.parse(input, pipelineP(using _), verboseFailures = true) match
      case Parsed.Success(passes, _) => Right(passes)
      case failure: Parsed.Failure   => Left(error(input, failure))

  private def pipelineP[$: P]: P[Seq[ModulePass]] = P(
    Start ~~ " ".repX ~~ passP.repX(sep = " ".repX ~~ "," ~~ " ".repX) ~~
      " ".repX ~~ End
  )

  private def passP[$: P]: P[ModulePass] =
    passOptionNameP./.flatMapX(name =>
      context.getPass(name) match
        case Some(pass) => pass.parse
        case None       =>
          Fail(
            s"'$name' does not refer to a registered pass.\n" +
              "Currently registered passes are:\n" +
              context.passContext.keys.toSeq.sorted.map(s => s"  - $s")
                .mkString("\n")
          )
    )

  private def error(input: String, failure: Parsed.Failure): String =
    s"$input\n${" " * failure.index}^\n${failure.label}"

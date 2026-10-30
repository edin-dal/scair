package scair.tools

import fastparse.*
import scair.MLContext
import scair.passes.mlir_opt.MLIROpt
import scair.transformations.*

/** Parses a pipeline directly into registered pass instances. */
final class PipelineParser(context: MLContext):

  def parse(input: String): Either[String, Seq[ModulePass]] =
    fastparse.parse(input, pipelineP(using _), verboseFailures = true) match
      case Parsed.Success(passes, _) => Right(passes)
      case failure: Parsed.Failure   => Left(error(failure))

  private def pipelineP[$: P]: P[Seq[ModulePass]] = P(
    Start ~~ " ".repX ~~ passP.repX(sep = " ".repX ~~ "," ~~ " ".repX) ~~
      " ".repX ~~ End
  )

  private def passP[$: P]: P[ModulePass] =
    passOptionNameP./.flatMapX(name =>
      val arguments: P[Option[Seq[String]]] =
        if name == "mlir-opt" then mlirPipelineP.?
        else
          P(!"[").opaque("`mlir-opt` to mark an MLIR pipeline").map(_ => None)
      arguments.flatMapX {
        case Some(values) => Pass(MLIROpt(arguments = values))
        case None         =>
          context.getPass(name) match
            case Some(pass) => pass.parse
            case None       =>
              Fail(
                s"'$name' does not refer to a registered pass.\n" +
                  "Currently registered passes are:\n" +
                  context.passContext.keys.toSeq.sorted.map(s => s"  - $s")
                    .mkString("\n")
              )
      }
    )

  private def mlirPipelineP[$: P]: P[Seq[String]] = P(
    ("["./ ~~ CharsWhile(_ != ']', 0).! ~~ "]").map(pipeline =>
      Seq(
        "--mlir-print-op-generic",
        "--allow-unregistered-dialect",
        "-p",
        s"builtin.module($pipeline)",
      )
    )
  )

  private def error(failure: Parsed.Failure): String =
    val input = failure.extra.input
    val position = input.prettyIndex(failure.index).split(":").map(_.toInt)
    val sourceLine = input.slice(0, input.length)
      .split("\n", -1)(position(0) - 1)
    val indicator = " " * (position(1) - 1) + "^"
    s"$sourceLine\n$indicator\n${failure.label}"

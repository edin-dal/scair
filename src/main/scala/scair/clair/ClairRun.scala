package scair.clair

import java.io.File
import scopt.OParser
import scala.io.Source
import scala.util.{Try, Success, Failure}
import fastparse.Parsed

case class Args(
    val input: Option[String] = None
)
object ClairRun {
  def main(args: Array[String]): Unit = {

    // Define CLI args
    val argbuilder = OParser.builder[Args]
    val argparser = {
      import argbuilder._
      OParser.sequence(
        programName("clair-run"),
        head("clair-run", "0"),
        // The input file - defaulting to stdin
        arg[String]("file")
          .optional()
          .text("input file")
          .action((x, c) => c.copy(input = Some(x)))
      )
    }

    // Parse the CLI args
    OParser.parse(argparser, args, Args()) match {
      case Some(args) =>
        // Open the input file or stdin
        val input = args.input match {
          case Some(file) => Source.fromFile(file)
          case None       => Source.stdin
        }

        // Parse content
        val parser = new ClairParser
        var module: Try[Parsed[DictType[String, DialectDef]]] = Try(
          parser.parseThis(
            input.mkString
          )
        )

        module match {
          case Success(v) =>
            v match {
              case fastparse.Parsed.Success(x, _) =>
                println(x)
              case e: fastparse.Parsed.Failure =>
                ClairExceptionMethods.throwParseError(input.toString, e)
            }
          case Failure(e) =>
            ClairExceptionMethods.throwCustomClairError(e)
        }
      case _ =>
    }
  }
}

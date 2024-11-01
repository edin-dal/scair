package scair.clair

import java.io.File
import scopt.OParser
import scala.io.Source
import scair.clair.ClairParser.dialectCTX

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
        val parser = scair.clair.ClairParser
        var module: DictType[String, DialectDef] = parser.parseThis(
          input.mkString,
          pattern = parser.EntryPoint(_)
        ) match {
          case fastparse.Parsed.Success(x, _) =>
            for dialect_def <- x.values do {
              println(dialect_def.print(0))
            }
            x
          case fastparse.Parsed.Failure(_, _, extra) =>
            sys.error(s"parse error:\n${extra.trace().longAggregateMsg}")
        }
      case _ =>
    }
  }
}

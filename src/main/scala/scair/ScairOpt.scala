import java.io.File
import scopt.OParser
import scala.io.Source
import scair.Printer
case class Args(
    val input: Option[String] = None
)
object ScairOpt {
  def main(args: Array[String]): Unit = {

    // Define CLI args
    val argbuilder = OParser.builder[Args]
    val argparser = {
      import argbuilder._
      OParser.sequence(
        programName("scair-opt"),
        head("scair-opt", "0"),
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
        val parser = new scair.Parser
        val module = parser.parseThis(
          input.mkString,
          pattern = parser.OperationPat(_)
        ) match {
          case fastparse.Parsed.Success(value, _) => value
          case fastparse.Parsed.Failure(_, _, extra) =>
            sys.error(s"parse error:\n${extra.trace().longAggregateMsg}")
        }

        // Print the parsed module if not errored
        val printer = new Printer()
        val output = printer.printOperation(module)
        println(output)

      case _ =>
    }
  }
}

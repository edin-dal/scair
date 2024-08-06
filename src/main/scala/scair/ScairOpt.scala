import java.io.File
import scopt.OParser
import scala.io.Source
import scair.Printer
case class Args(
    val input: Option[String] = None,
    val skip_verify: Boolean = true
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
          .action((x, c) => c.copy(input = Some(x))),
        opt[Unit]('s', "skip_verify")
          .optional()
          .text("Skip verification")
          .action((_, c) => c.copy(skip_verify = true))
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

        val skip_verify = args.skip_verify

        // Parse content
        val parser = new scair.Parser
        val module = parser.parseThis(
          input.mkString,
          pattern = parser.TopLevel(_)
        ) match {
          case fastparse.Parsed.Success(value, _) => value
          case fastparse.Parsed.Failure(_, _, extra) =>
            sys.error(s"parse error:\n${extra.trace().longAggregateMsg}")
        }

        // verify parsed content
        skip_verify match {
          case true  =>
          case false => value.verify()
        }

        // Print the parsed module if not errored
        val printer = new Printer()
        val output = printer.printOperation(module)
        println(output)

      case _ =>
    }
  }
}

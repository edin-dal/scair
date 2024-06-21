import java.io.File
import scopt.OParser
import scala.io.Source
case class Args(
    val input: Option[String] = None
)
object ScairOpt {
  def main(args: Array[String]): Unit = {

    val argbuilder = OParser.builder[Args]
    val argparser = {
      import argbuilder._
      OParser.sequence(
        programName("scair-opt"),
        head("scair-opt", "0"),
        arg[String]("<file>")
          .optional()
          .text("input file")
      )
    }

    OParser.parse(argparser, args, Args()) match {
      case Some(args) =>
        val input = args.input match {
          case Some(file) => Source.fromFile(file)
          case None       => Source.stdin
        }
        val parser = new scair.Parser
        val module = parser.parseThis(
          input.mkString,
          pattern = parser.OperationPat(_)
        ) match {
          case fastparse.Parsed.Success(value, _) => value
          case fastparse.Parsed.Failure(_, _, extra) =>
            sys.error(s"parse error:\n${extra.trace().longAggregateMsg}")
        }
        val output = scair.Printer.printOperation(module)
        println(output)
      case _ =>
    }
  }
}

import java.io.File
import scopt.OParser
import scala.io.Source
import scair.{Printer, Operation}
import scair.transformations.TransformContext
import scair.dialects.builtin.ModuleOp

case class Args(
    val input: Option[String] = None,
    val skip_verify: Boolean = true,
    val passes: Seq[String] = Seq()
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
          .action((_, c) => c.copy(skip_verify = true)),
        opt[Seq[String]]('p', "passes")
          .optional()
          .text("Specify passes to apply to the IR")
          .action((x, c) => c.copy(passes = x))
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

        val passes = args.passes

        // Parse content
        val parser = new scair.Parser
        var module: Operation = parser.parseThis(
          input.mkString,
          pattern = parser.TopLevel(_)
        ) match {
          case fastparse.Parsed.Success(value, _) => value
          case fastparse.Parsed.Failure(_, _, extra) =>
            sys.error(s"parse error:\n${extra.trace().longAggregateMsg}")
        }

        // verify parsed content
        if (!skip_verify) module.verify()

        // apply the specified passes
        if (passes.length != 0) {
          val transformCtx = new TransformContext()
          for (name <- passes) {
            transformCtx.getPass(name) match {
              case Some(pass) =>
                module = pass.transform(module)
                if (!skip_verify) module.verify()
              case None =>
            }
          }
        }

        // Print the parsed module if not errored
        val printer = new Printer()
        val output = module match {
          case x: ModuleOp =>
            ModuleOp.print(x, printer) // printer.printOperation(x)
          case _ =>
            throw new Exception(
              "Top level module must be the Builtin module of type ModuleOp.\n" +
                "==------------------==" +
                s"Check your tranformations: ${passes.mkString(", ")}" +
                "==------------------=="
            )
        }
        println(output)

      case _ =>
    }
  }
}

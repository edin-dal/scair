package scair

import java.io.File
import scopt.OParser
import scala.io.Source
import scair.{Printer, Operation}
import scair.transformations.TransformContext
import scair.dialects.builtin.ModuleOp

case class Args(
    val input: Option[String] = None,
    val skip_verify: Boolean = false,
    val split_input_file: Boolean = false,
    val print_generic: Boolean = false,
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
        opt[Unit]("split_input_file")
          .optional()
          .text("Split input file on `// -----`")
          .action((_, c) => c.copy(split_input_file = true)),
        opt[Unit]('g', "print_generic")
          .optional()
          .text("Print Strictly in Generic format")
          .action((_, c) => c.copy(print_generic = true)),
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

        val print_generic = args.print_generic

        val passes = args.passes
        // TODO: more robust separator splitting
        val input_chunks =
          if (args.split_input_file) input.mkString.split("\n// -----\n")
          else Array(input.mkString)

        val output_chunks = for (chunk <- input_chunks) yield {

          // Parse content
          val parser = new scair.Parser(args)
          val input_module = parser.parseThis(
            chunk,
            pattern = parser.TopLevel(_)
          ) match {
            case fastparse.Parsed.Success(value, _) => value
            case failure: fastparse.Parsed.Failure =>
              parser.error(failure)
          }

          // verify parsed content
          if (!skip_verify) input_module.verify()

          // apply the specified passes
          val transformCtx = new TransformContext()
          var module = input_module
          for (name <- passes) {
            transformCtx.getPass(name) match {
              case Some(pass) =>
                module = pass.transform(module)
                if (!skip_verify) module.verify()
              case None =>
            }
          }

          val printer = new Printer(print_generic)
          module match {
            case x: ModuleOp =>
              printer.printOperation(x) // printer.printOperation(x)
            case _ =>
              throw new Exception(
                "Top level module must be the Builtin module of type ModuleOp.\n" +
                  "==------------------==" +
                  s"Check your tranformations: ${passes.mkString(", ")}" +
                  "==------------------=="
              )
          }
        }

        // Print the processed modules if not errored
        println(output_chunks.mkString("\n// -----\n"))

      case _ =>
    }
  }
}

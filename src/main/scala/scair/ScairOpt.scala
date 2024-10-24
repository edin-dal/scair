import java.io.File
import scopt.OParser
import scala.io.Source
import scair.{Printer, Operation}
import scair.transformations.TransformContext
import scair.dialects.builtin.ModuleOp

package scair {

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

          val input_chunks = // TODO: more robust separator splitting
            if (args.split_input_file) input.mkString.split("\n// -----\n")
            else Array(input.mkString)

          // Parse content
          val modules: Seq[Operation] =
            for (chunk <- input_chunks)
              yield {
                val parser = new scair.Parser(args)
                parser.parseThis(
                  chunk,
                  pattern = parser.TopLevel(_)
                ) match {
                  case fastparse.Parsed.Success(value, _) => value
                  case failure: fastparse.Parsed.Failure =>
                    parser.error(failure)
                }
              }

          // verify parsed content
          if (!skip_verify) for (module <- modules) yield module.verify()

          // apply the specified passes
          val transformCtx = new TransformContext()
          val processed_modules = for (module <- modules) yield
            var mod = module
            for (name <- passes) {
              transformCtx.getPass(name) match {
                case Some(pass) =>
                  mod = pass.transform(module)
                  if (!skip_verify) module.verify()
                case None =>
              }
            }
            mod

          // Print the parsed module if not errored
          val outputs = for (module <- processed_modules) yield
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
          println(outputs.mkString("\n// -----\n"))

        case _ =>
      }
    }
  }

}

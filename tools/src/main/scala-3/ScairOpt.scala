import scair.MLContext
import scair.Printer
import scair.core.utils.Args
import scair.dialects.builtin.ModuleOp
import scair.ir.*
import scair.TransformContext
import scair.utils.allDialects
import scair.utils.allPasses
import scopt.OParser

import scala.io.Source
import scala.util.Failure
import scala.util.Success
import scala.util.Try

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
        opt[Unit]('a', "allow-unregistered-dialect")
          .optional()
          .text(
            "Accept unregistered operations and attributes, best effort with generic syntax."
          )
          .action((_, c) => c.copy(allow_unregistered = true)),
        opt[Unit]('s', "skip_verify")
          .optional()
          .text("Skip verification")
          .action((_, c) => c.copy(skip_verify = true)),
        opt[Unit]("split_input_file")
          .optional()
          .text("Split input file on `// -----`")
          .action((_, c) => c.copy(split_input_file = true)),
        opt[Unit]("parsing_diagnostics")
          .optional()
          .text(
            "Parsing diagnose mode, i.e parse errors are not fatal for the whole run"
          )
          .action((_, c) => c.copy(parsing_diagnostics = true)),
        opt[Unit]('g', "print_generic")
          .optional()
          .text("Print Strictly in Generic format")
          .action((_, c) => c.copy(print_generic = true)),
        opt[Seq[String]]('p', "passes")
          .optional()
          .text("Specify passes to apply to the IR")
          .action((x, c) => c.copy(passes = x)),
        opt[Unit]("verify_diagnostics")
          .optional()
          .text(
            "Verification diagnose mode, i.e verification errors are not fatal for the whole run"
          )
          .action((_, c) => c.copy(verify_diagnostics = true))
      )
    }

    // Parse the CLI args
    OParser.parse(argparser, args, Args()) match {
      case Some(args) =>
        import MyExtensions._
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
          val ctx = MLContext()
          ctx.register_all_dialects()
          val parser = new scair.Parser(ctx, args)
          parser.parseThis(
            chunk,
            pattern = parser.TopLevel(_)
          ) match {
            case fastparse.Parsed.Success(input_module, _) =>
              val processed_module = {
                var module = input_module
                // verify parsed content
                Try(if (!skip_verify) module.verify()) match {
                  case Success(_) =>
                    // apply the specified passes
                    val transformCtx = new TransformContext()
                    transformCtx.register_all_passes()
                    for (name <- passes) {
                      transformCtx.getPass(name) match {
                        case Some(pass) =>
                          module = pass.transform(module)
                          if (!skip_verify) module.verify()
                        case None =>
                      }
                    }
                    module
                  case Failure(exception) =>
                    if (args.verify_diagnostics) {
                      exception.getMessage
                    } else {
                      throw exception
                    }
                }

              }

              val printer = new Printer(print_generic)
              processed_module match {
                case output: String =>
                  output
                case x: ModuleOp =>
                  printer.printOperation(x)
                case _ =>
                  throw new Exception(
                    "Top level module must be the Builtin module of type ModuleOp.\n" +
                      "==------------------==" +
                      s"Check your tranformations: ${passes.mkString(", ")}" +
                      "==------------------=="
                  )
              }
            case failure: fastparse.Parsed.Failure =>
              parser.error(failure)

          }
        }

        // Print the processed modules if not errored
        println(output_chunks.mkString("\n// -----\n"))

      case _ =>
    }
  }

  object MyExtensions {
    extension (ctx: MLContext)
      def register_all_dialects(): Unit = {
        for (dialect <- allDialects) {
          ctx.registerDialect(dialect)
        }
      }

    extension (ctx: TransformContext)
      def register_all_passes(): Unit = {
        for (pass <- allPasses) {
          ctx.passContext += pass.name -> pass
        }
      }
  }
}

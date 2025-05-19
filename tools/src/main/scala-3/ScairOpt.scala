import scair.MLContext
import scair.Printer
import scair.TransformContext
import scair.core.utils.Args
import scair.dialects.builtin.ModuleOp
import scair.exceptions.VerifyException
import scair.ir.*
import scair.utils.allDialects
import scair.utils.allPasses
import scopt.OParser

import scala.io.Source

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
            "Accept unregistered operations and attributes, bestPRINT effort with generic syntax."
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
    val parsed_args = OParser.parse(argparser, args, Args()).get

    import MyExtensions._
    // Open the input file or stdin
    val input = parsed_args.input match {
      case Some(file) => Source.fromFile(file)
      case None       => Source.stdin
    }

    val skip_verify = parsed_args.skip_verify

    val print_generic = parsed_args.print_generic

    val passes = parsed_args.passes
    // TODO: more robust separator splitting
    val input_chunks =
      if (parsed_args.split_input_file) input.mkString.split("\n// -----\n")
      else Array(input.mkString)

    // Parse content
    val ctx = MLContext()
    ctx.register_all_dialects()

    input_chunks.foreach(chunk => {

      val parser = new scair.Parser(ctx, parsed_args)
      val printer = new Printer(print_generic)

      val input_module = parser.parseThis(
        chunk,
        pattern = parser.TopLevel(using _)
      ) match {
        case fastparse.Parsed.Success(input_module, _) =>
          Right(input_module)
        case failure: fastparse.Parsed.Failure =>
          Left(parser.error(failure))
      }

      if (!parsed_args.parsing_diagnostics && input_module.isLeft) then
        throw new Exception(input_module.left.get)

      val processed_module: Either[String, Operation] =
        input_module.flatMap(input_module => {
          var module = input_module
          // verify parsed content
          if (!skip_verify) module.verify() match {
            case Right(op) =>
              // apply the specified passes
              val transformCtx = new TransformContext()
              transformCtx.register_all_passes()
              for (name <- passes) {
                transformCtx.getPass(name) match {
                  case Some(pass) =>
                    module = pass.transform(module)
                    module = module.verify() match {
                      case Right(op) => op
                      case Left(errorMsg) =>
                        throw new VerifyException(errorMsg)
                    }
                  case None =>
                }
              }
              Right(module)
            case Left(errorMsg) =>
              if (parsed_args.verify_diagnostics) {
                Left(errorMsg)
              } else {
                throw new VerifyException(errorMsg)
              }
          }
          else {
            val transformCtx = new TransformContext()
            transformCtx.register_all_passes()
            for (name <- passes) {
              transformCtx.getPass(name) match {
                case Some(pass) =>
                  module = pass.transform(module)
                  module.verify() match {
                    case Right(_) =>
                    case Left(errorMsg) =>
                      throw new VerifyException(errorMsg)
                  }
                case None =>
              }
            }
            Right(module)
          }
        })

      processed_module match {
        case Left(errorMsg) =>
          printer.print(errorMsg)
        case Right(x) =>
          printer.print(x)(using 0)
      }

      if chunk != input_chunks.last then printer.print("// -----\n")
      printer.flush()

    })
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
          ctx.registerPass(pass)
        }
      }

  }

}

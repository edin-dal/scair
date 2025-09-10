package scair.tools

import scair.MLContext
import scair.Printer
import scair.TransformContext
import scair.core.utils.Args
import scair.exceptions.VerifyException
import scair.ir.*
import scopt.OParser

import scala.io.Source

trait ScairOptBase {
  val ctx = MLContext()
  val transformCtx = TransformContext()

  register_all_dialects()
  register_all_passes()

  def allDialects = {
    scair.utils.allDialects
  }

  def allPasses = {
    scair.utils.allPasses
  }

  final def register_all_dialects(): Unit = {
    for (dialect <- allDialects) {
      ctx.registerDialect(dialect)
    }
  }

  final def register_all_passes(): Unit = {
    for (pass <- allPasses) {
      transformCtx.registerPass(pass)
    }
  }

  def argParser : OParser[Unit, Args] =
    // Define CLI args
    val argbuilder = OParser.builder[Args]
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
      opt[Unit]('s', "skip-verify")
        .optional()
        .text("Skip verification")
        .action((_, c) => c.copy(skip_verify = true)),
      opt[Unit]("split-input-file")
        .optional()
        .text("Split input file on `// -----`")
        .action((_, c) => c.copy(split_input_file = true)),
      opt[Unit]("parsing-diagnostics")
        .optional()
        .text(
          "Parsing diagnose mode, i.e parse errors are not fatal for the whole run"
        )
        .action((_, c) => c.copy(parsing_diagnostics = true)),
      opt[Unit]('g', "print-generic")
        .optional()
        .text("Print Strictly in Generic format")
        .action((_, c) => c.copy(print_generic = true)),
      opt[Seq[String]]('p', "passes")
        .optional()
        .text("Specify passes to apply to the IR")
        .action((x, c) => c.copy(passes = x)),
      opt[Unit]("verify-diagnostics")
        .optional()
        .text(
          "Verification diagnose mode, i.e verification errors are not fatal for the whole run"
        )
        .action((_, c) => c.copy(verify_diagnostics = true)))
      
  def parse(ctx: MLConstext, args: Args, input: String): Either[String, Operation] = {
    val parser = new scair.Parser(ctx, args)
    parser.parseThis(
      input,
      pattern = parser.TopLevel(using _)
    ) match {
      case fastparse.Parsed.Success(input_module, _) =>
        Right(input_module)
      case failure: fastparse.Parsed.Failure =>
        Left(parser.error(failure))
    }
  }
    

  def run(args: Array[String]): Unit = {


    // Parse the CLI args
    val parsed_args = OParser.parse(argParser, args, Args()).get

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

    input_chunks.foreach(chunk => {

      val input_module = parse(ctx, parsed_args, chunk)

      if (!parsed_args.parsing_diagnostics && input_module.isLeft) then
        throw new Exception(input_module.left.get)

      val processed_module: Either[String, Operation] =
        input_module.flatMap(input_module => {
          var module =
            if (skip_verify) then Right(input_module)
            else input_module.structured.flatMap(_.verify())
          // verify parsed content
          module match {
            case Right(op) =>
              // apply the specified passes
              passes
                .map(transformCtx.getPass(_).get)
                .foldLeft(module)((module, pass) => {
                  module.map(pass.transform)
                })
            case Left(errorMsg) =>
              if (parsed_args.verify_diagnostics) {
                Left(errorMsg)
              } else {
                throw new VerifyException(errorMsg)
              }
          }
        })

      {
        val printer = new Printer(print_generic)
        processed_module.fold(
          printer.print,
          printer.printTopLevel
        )
        if chunk != input_chunks.last then printer.print("// -----\n")
        printer.flush()
      }
    })
  }

}

object ScairOpt extends ScairOptBase:
  def main(args: Array[String]): Unit = run(args)

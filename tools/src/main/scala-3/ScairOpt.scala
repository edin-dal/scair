package scair.tools

import scair.MLContext
import scair.Printer
import scair.exceptions.VerifyException
import scair.ir.*
import scopt.OParser

import scala.io.BufferedSource
import scala.io.Source

abstract class ScairToolBase[Args]:
  val ctx = MLContext()

  registerDialects()
  registerPasses()

  def dialects =
    scair.utils.allDialects

  def passes =
    scair.passes.allPasses

  final def registerDialects(): Unit =
    dialects.foreach(ctx.registerDialect)

  final def registerPasses(): Unit =
    passes.foreach(ctx.registerPass)

  final def commonHeaders =
    val argbuilder = OParser.builder[Args]

    OParser.sequence(
      argbuilder.programName(toolName),
      argbuilder.head(toolName, "0")
    )

  def parseArgs(args: Array[String]): Args
  def toolName: String
  def parse(args: Args)(input: BufferedSource): Array[Either[String, Operation]]

case class ScairOptArgs(
    val allow_unregistered: Boolean = false,
    val input: Option[String] = None,
    val skip_verify: Boolean = false,
    val split_input_file: Boolean = false,
    val parsing_diagnostics: Boolean = false,
    val print_generic: Boolean = false,
    val passes: Seq[String] = Seq(),
    val verify_diagnostics: Boolean = false
)

trait ScairOptBase extends ScairToolBase[ScairOptArgs]:

  override def parse(args: ScairOptArgs)(
      input: BufferedSource
  ): Array[Either[String, Operation]] =
    // TODO: more robust separator splitting
    val input_chunks =
      if args.split_input_file then input.mkString.split("\n// -----\n")
      else Array(input.mkString)
    input_chunks.map(input =>
      // Parse content
      val parser = new scair.Parser(
        ctx,
        inputPath = args.input,
        parsingDiagnostics = args.parsing_diagnostics,
        allowUnregisteredDialect = args.allow_unregistered
      )
      val parsedModule = parser.parseThis(
        input,
        pattern = parser.TopLevel(using _)
      ) match
        case fastparse.Parsed.Success(input_module, _) =>
          Right(input_module)
        case failure: fastparse.Parsed.Failure =>
          Left(parser.error(failure))

      if !args.parsing_diagnostics then
        parsedModule match
          case Left(msg)     => throw Exception(msg)
          case Right(module) => ()

      parsedModule
    )

  override def parseArgs(args: Array[String]): ScairOptArgs =
    // Define CLI args
    val argbuilder = OParser.builder[ScairOptArgs]
    val argparser =
      import argbuilder.*
      OParser.sequence(
        commonHeaders,
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
          .action((_, c) => c.copy(verify_diagnostics = true))
      )

    // Parse the CLI args
    OParser.parse(argparser, args, ScairOptArgs()).get

  def main(args: Array[String]): Unit =

    val parsed_args = parseArgs(args)

    // Open the input file or stdin
    val input = parsed_args.input match
      case Some(file) => Source.fromFile(file)
      case None       => Source.stdin

    val inputModules = parse(parsed_args)(input)

    inputModules.foreach(inputModule =>

      val processed_module: Either[String, Operation] =
        inputModule.flatMap(inputModule =>
          var module =
            if parsed_args.skip_verify then Right(inputModule)
            else inputModule.structured.flatMap(_.verify())
          // verify parsed content
          module match
            case Right(op) =>
              // apply the specified passes
              parsed_args.passes
                .map(ctx.getPass(_).get)
                .foldLeft(module)((module, pass) => module.map(pass.transform))
            case Left(errorMsg) =>
              if parsed_args.verify_diagnostics then Left(errorMsg)
              else throw new VerifyException(errorMsg)
        )

      {
        val printer = new Printer(parsed_args.print_generic)
        processed_module.fold(
          printer.print,
          printer.printTopLevel
        )
        if inputModule != inputModules.last then printer.print("// -----\n")
        printer.flush()
      }
    )

object ScairOpt extends ScairOptBase:
  def toolName: String = "scair-opt"

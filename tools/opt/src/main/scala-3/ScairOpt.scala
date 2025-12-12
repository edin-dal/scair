package scair.tools.opt

import scair.Printer
import scair.exceptions.VerifyException
import scair.ir.*
import scair.parse.*
import scair.tools.ScairToolBase
import scair.utils.OK
import scopt.OParser

import scala.io.BufferedSource
import scala.io.Source

//
// ░██████╗ ░█████╗░ ░█████╗░ ██╗ ██████╗░
// ██╔════╝ ██╔══██╗ ██╔══██╗ ██║ ██╔══██╗
// ╚█████╗░ ██║░░╚═╝ ███████║ ██║ ██████╔╝
// ░╚═══██╗ ██║░░██╗ ██╔══██║ ██║ ██╔══██╗
// ██████╔╝ ╚█████╔╝ ██║░░██║ ██║ ██║░░██║
// ╚═════╝░ ░╚════╝░ ╚═╝░░╚═╝ ╚═╝ ╚═╝░░╚═╝
//
// ░█████╗░ ██████╗░ ████████╗
// ██╔══██╗ ██╔══██╗ ╚══██╔══╝
// ██║░░██║ ██████╔╝ ░░░██║░░░
// ██║░░██║ ██╔═══╝░ ░░░██║░░░
// ╚█████╔╝ ██║░░░░░ ░░░██║░░░
// ░╚════╝░ ╚═╝░░░░░ ░░░╚═╝░░░
//

case class ScairOptArgs(
    val allowUnregistered: Boolean = false,
    val input: Option[String] = None,
    val skipVerify: Boolean = false,
    val splitInputFile: Boolean = false,
    val parsingDiagnostics: Boolean = false,
    val printGeneric: Boolean = false,
    val passes: Seq[String] = Seq(),
    val verifyDiagnostics: Boolean = false,
)

trait ScairOptBase extends ScairToolBase[ScairOptArgs]:

  override def dialects = scair.dialects.allDialects

  override def passes = scair.passes.allPasses

  override def parse(args: ScairOptArgs)(
      input: BufferedSource
  ): Array[OK[Operation]] =
    // TODO: more robust separator splitting
    val inputChunks =
      if args.splitInputFile then input.mkString.split("\n// -----\n")
      else Array(input.mkString)
    var indexOffset = 0
    inputChunks.map(input =>
      // Parse content
      val parser = new Parser(
        ctx,
        inputPath = args.input,
        parsingDiagnostics = args.parsingDiagnostics,
        allowUnregisteredDialect = args.allowUnregistered,
      )
      val parsed = parser.parse(
        input,
        parser = topLevelP(using _, parser),
      ) match
        case fastparse.Parsed.Success(inputModule, _) =>
          Right(inputModule)
        case failure: fastparse.Parsed.Failure =>
          Left(parser.error(failure, indexOffset))

      indexOffset += input.count(_ == '\n') + 2

      parsed
    )

  override def parseArgs(args: Array[String]): ScairOptArgs =
    // Define CLI args
    val argbuilder = OParser.builder[ScairOptArgs]
    val argparser =
      import argbuilder.*
      OParser.sequence(
        commonHeaders,
        // The input file - defaulting to stdin
        arg[String]("file").optional().text("input file")
          .action((x, c) => c.copy(input = Some(x))),
        opt[Unit]('a', "allow-unregistered-dialect").optional().text(
          "Accept unregistered operations and attributes, bestPRINT effort with generic syntax."
        ).action((_, c) => c.copy(allowUnregistered = true)),
        opt[Unit]('s', "skip-verify").optional().text("Skip verification")
          .action((_, c) => c.copy(skipVerify = true)),
        opt[Unit]("split-input-file").optional()
          .text("Split input file on `// -----`")
          .action((_, c) => c.copy(splitInputFile = true)),
        opt[Unit]("parsing-diagnostics").optional().text(
          "Parsing diagnose mode, i.e parse errors are not fatal for the whole run"
        ).action((_, c) => c.copy(parsingDiagnostics = true)),
        opt[Unit]('g', "print-generic").optional()
          .text("Print Strictly in Generic format")
          .action((_, c) => c.copy(printGeneric = true)),
        opt[Seq[String]]('p', "passes").optional()
          .text("Specify passes to apply to the IR")
          .action((x, c) => c.copy(passes = x)),
        opt[Unit]("verify-diagnostics").optional().text(
          "Verification diagnose mode, i.e verification errors are not fatal for the whole run"
        ).action((_, c) => c.copy(verifyDiagnostics = true)),
      )

    // Parse the CLI args
    OParser.parse(argparser, args, ScairOptArgs()).get

  def main(args: Array[String]): Unit =

    val parsedArgs = parseArgs(args)

    // Open the input file or stdin
    val input = parsedArgs.input match
      case Some(file) => Source.fromFile(file)
      case None       => Source.stdin

    val parsedModules = parse(parsedArgs)(input)

    parsedModules.foreach(parsedModule =>

      parsedModule match
        case Right(inputModule) =>
          val processedModule: OK[Operation] =
            var module =
              if parsedArgs.skipVerify then Right(inputModule)
              else inputModule.structured.flatMap(_.verify())
            // verify parsed content
            module match
              case Right(op) =>
                // apply the specified passes
                parsedArgs.passes.map(ctx.getPass(_).get).foldLeft(module)(
                  (module, pass) => module.map(pass.transform)
                )
              case Left(errorMsg) =>
                if parsedArgs.verifyDiagnostics then Left(errorMsg + "\n")
                else throw new VerifyException(errorMsg)

          {
            val printer = new Printer(parsedArgs.printGeneric)
            processedModule.fold(
              printer.print,
              printer.printTopLevel,
            )
            printer.flush()
          }
        case Left(errorMsg) =>
          if parsedArgs.parsingDiagnostics then println(errorMsg)
          else throw new Exception(errorMsg)

      if parsedModule != parsedModules.last then println("// -----")
    )

object ScairOpt extends ScairOptBase:
  def toolName: String = "scair-opt"

package scair.tools.runTool

import scair.dialects.builtin.ModuleOp
import scair.interpreter.Interpreter
import scair.interpreter.RuntimeCtx
import scair.ir.*
import scair.tools.ScairToolBase
import scopt.OParser

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
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
// ██████╗░ ██╗░░░██╗ ███╗░░██╗
// ██╔══██╗ ██║░░░██║ ████╗░██║
// ██████╔╝ ██║░░░██║ ██╔██╗██║
// ██╔══██╗ ██║░░░██║ ██║╚████║
// ██║░░██║ ╚██████╔╝ ██║░╚███║
// ╚═╝░░╚═╝ ░╚═════╝░ ╚═╝░░╚══╝
//

case class ScairRunArgs(
    val input: Option[String] = None
)

trait ScairRunBase extends ScairToolBase[ScairRunArgs]:

  override def dialects = scair.dialects.allDialects

  override def parseArgs(args: Array[String]): ScairRunArgs =
    // Define CLI args
    val argbuilder = OParser.builder[ScairRunArgs]
    val argparser =
      import argbuilder.*
      OParser.sequence(
        commonHeaders,
        // The input file - defaulting to stdin
        arg[String]("file")
          .optional()
          .text("input file")
          .action((x, c) => c.copy(input = Some(x)))
      )

    // Parse the CLI args
    OParser.parse(argparser, args, ScairRunArgs()).get

  override def parse(args: ScairRunArgs)(
      input: BufferedSource
  ): Array[Either[String, Operation]] =
    // Parse content
    // ONE CHUNK ONLY

    val input_module =
      val parser = new scair.Parser(ctx, inputPath = args.input)
      parser.parseThis(
        input.mkString,
        pattern = parser.TopLevel(using _)
      ) match
        case fastparse.Parsed.Success(input_module, _) =>
          Right(input_module)
        case failure: fastparse.Parsed.Failure =>
          Left(parser.error(failure))
    Array(input_module)

  def main(args: Array[String]): Unit =

    val parsed_args = parseArgs(args)

    // Open the input file or stdin
    val input = parsed_args.input match
      case Some(file) => Source.fromFile(file)
      case None       => Source.stdin

    // casted as moduleOp
    val module = parse(parsed_args)(input).head.right.get.asInstanceOf[ModuleOp]

    val module_block = module.body.blocks.head

    val interpreter = new Interpreter()
    var runtimeCtx =
      new RuntimeCtx(mutable.Map(), ListBuffer(), None)

    interpreter.register_implementations()

    val output = interpreter.interpret(module_block, runtimeCtx)

    if output.isDefined then
      if output.get == 1 then println("true")
      else if output.get == 0 then println("false")
      else println(output.get)

object ScairRun extends ScairRunBase:
  def toolName = "scair-run"

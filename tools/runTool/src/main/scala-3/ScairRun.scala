package scair.tools.runTool

import scair.dialects.builtin.ModuleOp
import scair.interpreter.Interpreter
import scair.interpreter.RuntimeCtx
import scair.ir.*
import scair.parse.*
import scair.tools.ScairToolBase
import scair.utils.*
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
        arg[String]("file").optional().text("input file")
          .action((x, c) => c.copy(input = Some(x))),
      )

    // Parse the CLI args
    OParser.parse(argparser, args, ScairRunArgs()).get

  override def parse(args: ScairRunArgs)(
      input: BufferedSource
  ): Array[OK[Operation]] =
    // Parse content
    // ONE CHUNK ONLY

    val inputModule =
      val parser = new Parser(ctx, inputPath = args.input)
      parser.parse(
        input = input.mkString,
        parser = moduleP(using _, parser),
      ) match
        case fastparse.Parsed.Success(inputModule, _) =>
          OK(inputModule)
        case failure: fastparse.Parsed.Failure =>
          Err(parser.error(failure))
    Array(inputModule)

  def main(args: Array[String]): Unit =

    val parsedArgs = parseArgs(args)

    // Open the input file or stdin
    val input = parsedArgs.input match
      case Some(file) => Source.fromFile(file)
      case None       => Source.stdin

    // casted as moduleOp
    val module = parse(parsedArgs)(input).head.get.asInstanceOf[ModuleOp]

    val moduleBlock = module.body.blocks.head

    val interpreter = new Interpreter()
    var runtimeCtx =
      new RuntimeCtx(mutable.Map(), ListBuffer(), None)

    interpreter.register_implementations()

    val output = interpreter.interpret(moduleBlock, runtimeCtx)

    if output.isDefined then
      if output.get == 1 then println("true")
      else if output.get == 0 then println("false")
      else println(output.get)

object ScairRun extends ScairRunBase:
  def toolName = "scair-run"

package scair.tools.runTool

import scair.dialects.builtin.ModuleOp
import scair.interpreter.Interpreter
import scair.ir.*
import scair.parse.*
import scair.tools.ScairToolBase
import scair.utils.*
import scair.verify.Verifier
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
// ██████╗░ ██╗░░░██╗ ███╗░░██╗
// ██╔══██╗ ██║░░░██║ ████╗░██║
// ██████╔╝ ██║░░░██║ ██╔██╗██║
// ██╔══██╗ ██║░░░██║ ██║╚████║
// ██║░░██║ ╚██████╔╝ ██║░╚███║
// ╚═╝░░╚═╝ ░╚═════╝░ ╚═╝░░╚══╝
//

case class ScairRunArgs(
    val input: Option[String] = None,
    skipVerify: Boolean = false,
)

trait ScairRunBase extends ScairToolBase[ScairRunArgs]:

  def interpreterDialects = scair.interpreter.allInterpreterDialects

  def entryPoint = "main"

  def verboseInterpreter = true

  def printScopes = false

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
      val parser = Parser(ctx, inputPath = args.input)
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

    val module = parse(parsedArgs)(input).head.get.asInstanceOf[ModuleOp]

    if !parsedArgs.skipVerify then
      Verifier.verify(module) match
        case e: scair.utils.Err => throw new Exception(e.msg)
        case _                  => ()

    // construct interpreter and runtime context
    val interpreter =
      new Interpreter(
        module,
        interpreterDialects,
      )

    // TODO: Allow specifying the function for entry point and default to main if not
    val output = interpreter
      .call_op(entryPoint, interpreter.create_scope(entryPoint), Seq())

    printScopes match
      case true =>
        interpreter.scopes.foreach(_.prettyPrint())
      case false => ()

    verboseInterpreter match
      case true =>
        output match
          case Seq()      => println("Result: ()")
          case Seq(value) =>
            print("Result: ")
            interpreter.interpreter_print(value)
          case multiple =>
            print("Result: (")
            multiple.foreach(res => print(s"$res, "))
            print(")")
      case false => ()

object ScairRun extends ScairRunBase:
  def toolName = "scair-run"

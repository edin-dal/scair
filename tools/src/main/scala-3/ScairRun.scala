package scair.tools

import scair.MLContext
import scair.AttrParser
import scair.TransformContext
import scair.core.utils.Args
import scair.ir.*
import scopt.OParser

import scala.io.Source
import scair.dialects.builtin.ModuleOp
import scair.dialects.builtin.IntegerAttr
import scair.dialects.builtin.IntegerType
import scair.dialects.func
import scair.dialects.arith

trait ScairRunBase {
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

  def run(args: Array[String]): Unit = {

    // Define CLI args
    val argbuilder = OParser.builder[Args]
    val argparser = {
      import argbuilder._
      OParser.sequence(
        programName("scair-run"),
        head("scair-run", "0"),
        // The input file - defaulting to stdin
        arg[String]("file")
          .optional()
          .text("input file")
          .action((x, c) => c.copy(input = Some(x)))
      )
    }

    // Parse the CLI args
    val parsed_args = OParser.parse(argparser, args, Args()).get

    // Open the input file or stdin
    val input = parsed_args.input match {
      case Some(file) => Source.fromFile(file)
      case None       => Source.stdin
    }

    // Parse content
    // ONE CHUNK ONLY

    val input_module = {
      val parser = new scair.Parser(ctx, parsed_args)
      parser.parseThis(
        input.mkString,
        pattern = parser.TopLevel(using _)
      ) match {
        case fastparse.Parsed.Success(input_module, _) =>
          Right(input_module)
        case failure: fastparse.Parsed.Failure =>
          Left(parser.error(failure))
      }
    }

    if (!parsed_args.parsing_diagnostics && input_module.isLeft) then
        throw new Exception(input_module.left.get)

    // casted as moduleOp
    val module = input_module.right.get.asInstanceOf[ModuleOp]

    // main block
    // TODO: handling multiple blocks, assuming only one for now
    val module_block = module.body.blocks.head

    val interpreter = new Interpreter()
    var interpreterCtx = new InterpreterCtx(Map(), Map(), List())

    //val interpreted_output = interpreter.interpret(main_block)
    //println(module_block.operations)

    // TODO: assumed one block only for now
    val output = interpreter.interpret(module_block, interpreterCtx)
    println(output)
  }

}

object ScairRun extends ScairRunBase:
  def main(args: Array[String]): Unit = run(args)

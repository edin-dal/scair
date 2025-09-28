package scair.tools

import scair.MLContext
import scair.TransformContext
import scair.core.utils.Args
import scair.ir.*
import scopt.OParser

import scala.io.Source
import scair.dialects.builtin.ModuleOp
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

    val module = input_module.right.get.asInstanceOf[ModuleOp]
    val main_op = module.body.blocks.head.operations(0).asInstanceOf[func.Func]

    // assuming main function only for now
    val main_body = main_op.body
    val return_val = main_body.blocks.head.operations.last.asInstanceOf[func.Return].operands.head.owner.getOrElse(0)
    val constant = return_val.asInstanceOf[arith.Constant].value
    println(constant)


    

    /*
     * NOTES:
     * needs interpreter class
     * some sort of table/dictionary for operations
     * main call_operation function (op, args) to kickstart interpretation
     *   
     * FLOW:
     * instantiate interpreter
     * register implementations
     * parsing already done
     * getting args already done?
     * interpreter call op function
     * get result
     */



  }

}

object ScairRun extends ScairRunBase:
  def main(args: Array[String]): Unit = run(args)

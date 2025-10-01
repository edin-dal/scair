package scair.tools

import scair.MLContext
import scair.AttrParser
import scair.Interpreter
import scair.TransformContext
import scair.core.utils.Args
import scair.ir.*
import scopt.OParser

import scala.io.Source
import scair.dialects.builtin.ModuleOp
import scair.dialects.builtin.IntegerAttr
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

  def interpret(op: Operation): Attribute = {
    var interpretedVal: Attribute = null
    op match {
      case constant: arith.Constant =>
        interpretedVal = constant.value
      // TODO: handling block arguments for all arithmetic operations
      case addIOp: arith.AddI =>
        val lhs = addIOp.lhs.owner.get
        val rhs = addIOp.rhs.owner.get
        (lhs, rhs) match {
          case (l: Operation, r: Operation) =>
            val interpretedL = interpret(l)
            val interpretedR = interpret(r)
            (interpretedL, interpretedR) match {
              case (l: IntegerAttr, r: IntegerAttr) =>
                interpretedVal = l + r
              case _ =>
                  throw new Exception("Unsupported operand types for AddI")
            }
          case _ =>
            throw new Exception("Expected operations as operands for AddI")
        }
      case subIOp: arith.SubI =>
        val lhs = subIOp.lhs.owner.get
        val rhs = subIOp.rhs.owner.get
        (lhs, rhs) match {
          case (l: Operation, r: Operation) =>
            val interpretedL = interpret(l)
            val interpretedR = interpret(r)
            (interpretedL, interpretedR) match {
              case (l: IntegerAttr, r: IntegerAttr) =>
                interpretedVal = l - r
              case _ =>
                  throw new Exception("Unsupported operand types for SubI")
            }
          case _ =>
            throw new Exception("Expected operations as operands for SubI")
        }
      case mulIOp: arith.MulI =>
        val lhs = mulIOp.lhs.owner.get
        val rhs = mulIOp.rhs.owner.get
        (lhs, rhs) match {
          case (l: Operation, r: Operation) =>
            val interpretedL = interpret(l)
            val interpretedR = interpret(r)
            (interpretedL, interpretedR) match {
              case (l: IntegerAttr, r: IntegerAttr) =>
                interpretedVal = l * r
              case _ =>
                  throw new Exception("Unsupported operand types for MulI")
            }
          case _ =>
            throw new Exception("Expected operations as operands for MulI")
        }
      // TODO: treating signed and unsigned division separately
      case divUIOp: arith.DivUI =>
        val lhs = divUIOp.lhs.owner.get
        val rhs = divUIOp.rhs.owner.get
        (lhs, rhs) match {
          case (l: Operation, r: Operation) =>
            val interpretedL = interpret(l)
            val interpretedR = interpret(r)
            (interpretedL, interpretedR) match {
              case (l: IntegerAttr, r: IntegerAttr) =>
                interpretedVal = l.divUI(r)
              case _ =>
                  throw new Exception("Unsupported operand types for DivUI")
            }
          case _ =>
            throw new Exception("Expected operations as operands for DivUI")
        }
      case divSIOp: arith.DivSI =>
        val lhs = divSIOp.lhs.owner.get
        val rhs = divSIOp.rhs.owner.get
        (lhs, rhs) match {
          case (l: Operation, r: Operation) =>
            val interpretedL = interpret(l)
            val interpretedR = interpret(r)
            (interpretedL, interpretedR) match {
              case (l: IntegerAttr, r: IntegerAttr) =>
                interpretedVal = l.divSI(r)
              case _ =>
                  throw new Exception("Unsupported operand types for DivSI")
            }
          case _ =>
            throw new Exception("Expected operations as operands for DivSI")
        }
    }
    return interpretedVal
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

    // main function
    val main_op = module.body.blocks.head.operations(0).asInstanceOf[func.Func]

    // assuming main function only for now
    // very basic return constant implementation
    // extend to evaluate other ops instead of just constant
    val main_return_op = main_op.body.blocks.head.operations.last.asInstanceOf[func.Return]
    val main_return_parent = main_return_op.operands.head.owner.getOrElse(0).asInstanceOf[Operation]

    val output = interpret(main_return_parent)
    println(output)

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

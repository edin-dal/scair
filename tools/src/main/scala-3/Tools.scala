package scair.tools

import scair.MLContext
import scair.ir.Dialect
import scair.ir.Operation
import scair.transformations.ModulePass
import scair.utils.OK
import scopt.OParser

import scala.io.BufferedSource

//
// ████████╗ ░█████╗░ ░█████╗░ ██╗░░░░░ ░██████╗
// ╚══██╔══╝ ██╔══██╗ ██╔══██╗ ██║░░░░░ ██╔════╝
// ░░░██║░░░ ██║░░██║ ██║░░██║ ██║░░░░░ ╚█████╗░
// ░░░██║░░░ ██║░░██║ ██║░░██║ ██║░░░░░ ░╚═══██╗
// ░░░██║░░░ ╚█████╔╝ ╚█████╔╝ ███████╗ ██████╔╝
// ░░░╚═╝░░░ ░╚════╝░ ░╚════╝░ ╚══════╝ ╚═════╝░
//

abstract class ScairToolBase[Args]:
  val ctx = MLContext()

  registerDialects()
  registerPasses()

  def dialects: Seq[Dialect] = Seq()

  def passes: Seq[MLContext => ModulePass] = Seq()

  final def registerDialects(): Unit =
    dialects.foreach(ctx.registerDialect)

  final def registerPasses(): Unit =
    passes.foreach(ctx.registerPass)

  final def commonHeaders =
    val argbuilder = OParser.builder[Args]

    OParser.sequence(
      argbuilder.programName(toolName),
      argbuilder.head(toolName, "0"),
    )

  def parseArgs(args: Array[String]): Args
  def toolName: String
  def parse(args: Args)(input: BufferedSource): Array[OK[Operation]]

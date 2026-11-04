package scair.passes.mlir_opt

import fastparse.Parsed
import scair.MLContext
import scair.ir.*
import scair.parse.Parser
import scair.print.AssemblyPrinter
import scair.transformations.*

import java.io.StringWriter
import java.nio.file.Files
import scala.io.Source
import scala.jdk.CollectionConverters.*
import scala.util.Using

/** Runs the operation through MLIR's `mlir-opt` and parses its output back.
  *
  * `mlir-opt[cse,canonicalize]` in a pipeline is shorthand for running the
  * given MLIR pass pipeline.
  *
  * @param executable
  *   The `mlir-opt` to run; `$SCAIR_MLIR_OPT` if set, else from the PATH.
  * @param generic
  *   Whether to hand the IR to `mlir-opt` in generic format.
  * @param arguments
  *   The command line arguments to `mlir-opt`.
  * @param allowUnregisteredDialect
  *   Whether to accept unregistered operations in `mlir-opt`'s output.
  */
final case class MLIROpt(
    executable: String = sys.env.getOrElse("SCAIR_MLIR_OPT", "mlir-opt"),
    generic: Boolean = true,
    arguments: Seq[String] = Seq(),
    allowUnregisteredDialect: Boolean = false,
) extends ModulePass
    with PassName["mlir-opt"] derives PassCompanion:

  override def transform(op: Operation)(using ctx: MLContext): Operation =
    val input = StringWriter()
    val printer = AssemblyPrinter(strictlyGeneric = generic, p = input)
    printer.printTopLevel(op)
    printer.flush()

    val command = executable +: arguments
    // Piping stdin can kill Scala Native with SIGPIPE if mlir-opt exits early.
    val inputFile = Files.createTempFile("scair-mlir-opt", ".mlir")
    val output =
      try
        Files.write(inputFile, input.toString.getBytes)
        val process =
          try
            ProcessBuilder(command.asJava).redirectInput(inputFile.toFile)
              .redirectError(ProcessBuilder.Redirect.INHERIT).start()
          catch
            case e: java.io.IOException =>
              throw PassFailure(s"cannot run '$executable': ${e.getMessage}")
        val output = Using
          .resource(Source.fromInputStream(process.getInputStream))(
            _.mkString
          )
        if process.waitFor() != 0 then
          throw PassFailure(s"'${command.mkString(" ")}' failed")
        output
      finally Files.delete(inputFile)

    val parser =
      Parser(
        ctx,
        inputPath = Some(s"<$executable output>"),
        parsingDiagnostics = true,
        allowUnregisteredDialect = allowUnregisteredDialect,
      )
    parser.parse(output) match
      case Parsed.Success(module, _) => module
      case failure: Parsed.Failure   =>
        throw PassFailure(
          s"cannot parse the output of '${command.mkString(" ")}':\n" +
            parser.error(failure)
        )

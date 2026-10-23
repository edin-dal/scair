package scair.transformations

import scair.MLContext
import scair.ir.*

// ██████╗░ ░█████╗░ ░██████╗ ░██████╗ ███████╗ ░██████╗
// ██╔══██╗ ██╔══██╗ ██╔════╝ ██╔════╝ ██╔════╝ ██╔════╝
// ██████╔╝ ███████║ ╚█████╗░ ╚█████╗░ █████╗░░ ╚█████╗░
// ██╔═══╝░ ██╔══██║ ░╚═══██╗ ░╚═══██╗ ██╔══╝░░ ░╚═══██╗
// ██║░░░░░ ██║░░██║ ██████╔╝ ██████╔╝ ███████╗ ██████╔╝
// ╚═╝░░░░░ ╚═╝░░╚═╝ ╚═════╝░ ╚═════╝░ ╚══════╝ ╚═════╝░

/** A transformation over an operation, usually a module.
  *
  * Passes are case classes whose fields are their options, and derive their
  * [[PassDefs]]:
  * {{{
  * final case class MyPass(maxIterations: Int = 10) extends ModulePass
  *     derives PassDefs
  * }}}
  */
trait ModulePass:
  def transform(op: Operation)(using MLContext): Operation

trait WalkerPass extends ModulePass:
  def walker(using MLContext): PatternRewriteWalker

  final override def transform(op: Operation)(using MLContext): Operation =
    walker.rewrite(op)
    return op

  final def transform(block: Block)(using MLContext): Block =
    walker.rewrite(block)
    return block

  final def transform(region: Region)(using MLContext): Region =
    walker.rewrite(region)
    return region

/** A one-shot dialect conversion, run as a pass.
  *
  * The type conversion and operation conversion patterns are applied in a
  * single sweep over the regions of the operation the pass is given; see
  * [[ConversionDriver]].
  */
trait ConversionPass extends ModulePass:

  def typeConverter: TypeConverter

  def patterns: Seq[ConversionPattern]

  /** Whether to fold the materialized casts once the conversion is done. */
  def reconcile: Boolean = true

  final override def transform(op: Operation)(using MLContext): Operation =
    ConversionDriver(typeConverter, patterns).convert(op)
    if reconcile then ReconcileCasts.walker.rewrite(op)
    return op

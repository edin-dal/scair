package scair.transformations

import scair.MLContext
import scair.ir.*

// ██████╗░ ░█████╗░ ░██████╗ ░██████╗ ███████╗ ░██████╗
// ██╔══██╗ ██╔══██╗ ██╔════╝ ██╔════╝ ██╔════╝ ██╔════╝
// ██████╔╝ ███████║ ╚█████╗░ ╚█████╗░ █████╗░░ ╚█████╗░
// ██╔═══╝░ ██╔══██║ ░╚═══██╗ ░╚═══██╗ ██╔══╝░░ ░╚═══██╗
// ██║░░░░░ ██║░░██║ ██████╔╝ ██████╔╝ ███████╗ ██████╔╝
// ╚═╝░░░░░ ╚═╝░░╚═╝ ╚═════╝░ ╚═════╝░ ╚══════╝ ╚═════╝░

abstract class ModulePass(ctx: MLContext):
  val name: String
  def transform(op: Operation): Operation = ???

abstract class WalkerPass(ctx: MLContext) extends ModulePass(ctx):
  def walker: PatternRewriteWalker

  final override def transform(op: Operation): Operation =
    walker.rewrite(op)
    return op

  final def transform(block: Block): Block =
    walker.rewrite(block)
    return block

  final def transform(region: Region): Region =
    walker.rewrite(region)
    return region

/** A one-shot dialect conversion, run as a pass.
  *
  * The type conversion and operation conversion patterns are applied in a
  * single sweep over the regions of the operation the pass is given; see
  * [[ConversionDriver]].
  */
abstract class ConversionPass(ctx: MLContext) extends ModulePass(ctx):

  def typeConverter: TypeConverter

  def patterns: Seq[ConversionPattern]

  /** Whether to fold the materialized casts once the conversion is done. */
  def reconcile: Boolean = true

  final override def transform(op: Operation): Operation =
    ConversionDriver(typeConverter, patterns).convert(op)
    if reconcile then ReconcileCasts.walker.rewrite(op)
    return op

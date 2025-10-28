package scair.transformations

import scair.MLContext
import scair.ir.*

// ██████╗░ ░█████╗░ ░██████╗ ░██████╗ ███████╗ ░██████╗
// ██╔══██╗ ██╔══██╗ ██╔════╝ ██╔════╝ ██╔════╝ ██╔════╝
// ██████╔╝ ███████║ ╚█████╗░ ╚█████╗░ █████╗░░ ╚█████╗░
// ██╔═══╝░ ██╔══██║ ░╚═══██╗ ░╚═══██╗ ██╔══╝░░ ░╚═══██╗
// ██║░░░░░ ██║░░██║ ██████╔╝ ██████╔╝ ███████╗ ██████╔╝
// ╚═╝░░░░░ ╚═╝░░╚═╝ ╚═════╝░ ╚═════╝░ ╚══════╝ ╚═════╝░

abstract class ModulePass(ctx: MLContext) {
  val name: String
  def transform(op: Operation): Operation = ???
}

abstract class WalkerPass(ctx: MLContext) extends ModulePass(ctx) {
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

}

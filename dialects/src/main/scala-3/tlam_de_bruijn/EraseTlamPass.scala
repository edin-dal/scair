package scair.passes

import scair.MLContext
import scair.ir.*
import scair.transformations.RewriteMethods
import scair.transformations.*
import scair.dialects.builtin.*
import scair.dialects.tlam_de_bruijn.*

final class EraseTLamPass(ctx: MLContext) extends ModulePass(ctx):
  override val name: String = "erase-tlam"

  override def transform(op: Operation): Operation =
    op match
      case m: ModuleOp =>
        eraseInModule(m); m
      case other => other

  private def eraseInModule(m: ModuleOp): Unit =
    def walkRegion(r: Region): Unit =
      r.blocks.foreach { b =>
        val ops = b.operations.toSeq
        ops.foreach {
          case tl: TLambda =>
            val bodyBlock = tl.tBody.blocks.head
            val bodyOps = bodyBlock.operations.toSeq

            val tret = bodyOps.last match
              case r: TReturn => r
              case _          => throw new Exception("TLambda without TReturn")

            val moved = bodyOps.dropRight(1).map(bodyBlock.detachOp)
            RewriteMethods.insertOpsBefore(tl, moved)

            RewriteMethods.replaceOp(
              tl,
              newOps = Seq.empty,
              newResults = Some(Seq(tret.value)),
            )

          case other =>
            other.regions.foreach(walkRegion)
        }
      }

    walkRegion(m.regions.head)

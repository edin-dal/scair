package scair.verify

import scair.analysis.DominanceInfo
import scair.ir.*
import scair.utils.Err
import scair.utils.OK

import scala.util.boundary
import scala.util.boundary.break

object SSADominanceCheck extends VerifierCheck:
  override val name: String = "ssa-dominance"

  override def run(root: Operation): OK[Unit] =
    val dom = new DominanceInfo(root)

    def walkRegion(r: Region): OK[Unit] =
      boundary[OK[Unit]] {
        r.blocks.foreach { b =>
          b.operations.foreach { op =>
            // Check operand dominance at this use site
            op.operands.foreach { v =>
              if !dom.valueDominates(v, op) then
                break(
                  Err(
                    s"value $v does not dominate its use in op `${op.name}`"
                  ): OK[Unit]
                )
            }

            // Recurse into nested regions
            op.regions.foreach { rr =>
              walkRegion(rr) match
                case e: Err => break(e: OK[Unit])
                case _      => ()
            }
          }
        }
        OK(())
      }

    boundary[OK[Unit]] {
      root.regions.foreach { r =>
        walkRegion(r) match
          case e: Err => break(e: OK[Unit])
          case _      => ()
      }
      OK(())
    }

package scair.verify

import scair.analysis.DominanceInfo
import scair.ir.*
import scair.utils.{Err, OK}
import scala.util.boundary, boundary.break

object SSADominanceCheck extends VerifierCheck:
  override val name: String = "ssa-dominance"

  override def run(root: Operation): OK[Unit] =
    val dom = new DominanceInfo(root)

    def walkOp(o: Operation): OK[Unit] =
      // Check operand dominance at this use site
      o.operands.foreach { v =>
        val vv = v.asInstanceOf[Value[Attribute]]
        if !dom.valueDominates(vv, o) then
          return Err(s"value $v does not dominate its use in op `${o.name}`")
      }

      // Recurse into nested regions
      o.regions.foreach { r =>
        walkRegion(r) match
          case e: Err => return e
          case _      => ()
      }
      OK(())

    def walkRegion(r: Region): OK[Unit] =
      boundary:
        r.blocks.foreach { b =>
          b.operations.foreach { op =>
            walkOp(op) match
              case e: Err => break(e)
              case _      => ()
          }
        }
        OK(())

    walkOp(root)

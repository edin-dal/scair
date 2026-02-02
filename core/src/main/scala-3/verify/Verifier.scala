package scair.verify

import scair.ir.*
import scair.utils.Err
import scair.utils.OK

import scala.util.boundary
import scala.util.boundary.break

object Verifier:

  val defaultChecks: Seq[VerifierCheck] = Seq(SSADominanceCheck)

  def verify(
      root: Operation,
      checks: Seq[VerifierCheck] = defaultChecks,
  ): OK[Operation] =
    root.verify() match
      case e: Err => e
      case _      =>
        boundary[OK[Operation]] {
          for chk <- checks do
            chk.run(root) match
              case e: Err => break(Err(s"${chk.name}: ${e.msg}"): OK[Operation])
              case _      => ()
          OK(root)
        }

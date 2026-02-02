package scair.verify

import scair.MLContext
import scair.ir.*
import scair.utils.Err
import scair.utils.OK

import scala.util.boundary
import scala.util.boundary.break

object Verifier:

  final case class Config(
      genericChecks: Seq[VerifierCheck] = Seq(SSADominanceCheck)
  )

  def verify(
      root: Operation,
      ctx: MLContext,
      cfg: Config = Config(),
  ): OK[Operation] =
    root.verify() match
      case e: Err => e
      case _      =>
        boundary[OK[Operation]] {
          for chk <- cfg.genericChecks do
            chk.run(root) match
              case e: Err => break(Err(s"${chk.name}: ${e.msg}"): OK[Operation])
              case _      => ()
          OK(root)
        }

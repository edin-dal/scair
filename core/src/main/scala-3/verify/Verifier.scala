package scair.verify

import scair.MLContext
import scair.ir.*
import scair.utils.{Err, OK}

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
        val allChecks = cfg.genericChecks ++ ctx.verifierRegistry.all
        allChecks.foreach { chk =>
          chk.run(root) match
            case e: Err => return Err(s"${chk.name}: ${e.msg}")
            case _      => ()
        }
        OK(root)

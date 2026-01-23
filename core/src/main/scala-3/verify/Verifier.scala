// core/src/main/scala/scair/verify/Verifier.scala
package scair.verify

import scair.MLContext
import scair.ir.*
import scair.utils.{Err, OK}

object Verifier:

  final case class Config(
      genericChecks: Seq[VerifierCheck] = Seq(SSADominanceCheck)
      // dialect checks come from ctx.verifierRegistry
  )

  def verify(
      root: Operation,
      ctx: MLContext,
      cfg: Config = Config(),
  ): OK[Operation] =
    // Phase 1: structural verification
    root.verify() match
      case e: Err => e
      case _      =>
        // Phase 2: generic checks (core)
        val allChecks = cfg.genericChecks ++ ctx.verifierRegistry.all
        allChecks.foreach { chk =>
          chk.run(root) match
            case e: Err => return Err(s"${chk.name}: ${e.msg}")
            case _      => ()
        }
        OK(root)

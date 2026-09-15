package scair.verify

import scair.analysis.DominanceInfo
import scair.ir.*
import scair.utils.Err
import scair.utils.OK

object SSADominanceCheck extends VerifierCheck:
  override val name: String = "ssa-dominance"

  override def run(root: Operation): OK[Unit] =
    val dom = new DominanceInfo(root)
    var error: Err = null

    // Check operand dominance at each use site.
    val check: Operation => WalkResult = op =>
      var failed: Value[Attribute] = null
      op.forEachOperand(v =>
        if (failed eq null) && !dom.valueDominates(v, op) then failed = v
      )
      if failed eq null then WalkResult.Advance
      else
        error = Err(
          s"value $failed does not dominate its use in op `${op.name}`"
        )
        WalkResult.Interrupt

    root.forEachRegion(region => if error eq null then region.walk(check))
    if error eq null then OK(()) else error

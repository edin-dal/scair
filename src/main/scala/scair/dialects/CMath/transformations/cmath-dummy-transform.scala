package scair.dialects.CMath.transformations.cdt

import scair.{Operation, MLContext}
import scair.transformations.{ModulePass}

object DummyPass extends ModulePass {
  override val name = "dummy-pass"

  override def transform(op: Operation): Operation = {
    return op
  }
}

package scair.transformations.samplemathcanon

import scair.dialects.samplemath.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.transformations.*
import scair.transformations.patterns.*

// c0 * c1 => c0 * c1 :sunglasses:
val MulMulConstant = pattern {
  case Mul(
        Owner(Constant(c0: IntegerAttr, _)),
        Owner(Constant(c1: IntegerAttr, _)),
        x
      ) =>
    val cv = Result(x.typ)
    Seq(Constant(c0 * c1, cv))
}

val DeadConstant = pattern {
  case Constant(_, value) if value.uses.isEmpty => PatternAction.Erase
}

object SampleMathCanon extends ModulePass {
  override val name = "sample-math-canon"

  override def transform(op: Operation): Operation = {
    val prw = new PatternRewriteWalker(
      GreedyRewritePatternApplier(Seq(MulMulConstant, DeadConstant))
    )
    prw.rewrite_op(op)
    return op
  }

}

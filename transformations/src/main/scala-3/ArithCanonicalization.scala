package scair.transformations.arith_canonicalization

import scair.dialects.arith.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.transformations.*
import scair.transformations.patterns.*

// TODO: Generalize in terms of effects and out of dialect
val RemoveUnusedOperations = pattern {
  case _: IsTerminator => PatternAction.Abort
  case op: NoMemoryEffect if op.results.forall(_.uses.isEmpty) => PatternAction.Erase
  case op: NoMemoryEffect => PatternAction.Abort
}

// TODO: Generalize in Commutative/ConstantLike
val AddICommute = pattern {
  case AddI(rhs = Owner(_: Constant)) => PatternAction.Abort
  case AddI(lhs @ Owner(Constant(value = c0: IntegerAttr)), rhs, Result(tpe)) =>
    AddI(rhs, lhs, Result(tpe))
}

// addi(addi(x, c0), c1) -> addi(x, c0 + c1)
val AddIAddConstant = pattern {
  case AddI(
        Owner(
          AddI(x @ Value(tpe), Owner(Constant(c0: IntegerAttr, _)), _)
        ),
        Owner(Constant(c1: IntegerAttr, _)),
        _
      ) =>
    val cv = Result(tpe)
    val c = Constant(c0 + c1, cv)
    Seq(c, AddI(x, cv, Result(tpe)))
}

object ArithCanonicalize extends ModulePass {
  override val name = "arith-canonicalize"

  override def transform(op: Operation): Operation = {
    val prw = new PatternRewriteWalker(
      GreedyRewritePatternApplier(
        Seq(
          RemoveUnusedOperations,
          AddICommute,
          AddIAddConstant
        )
      )
    )
    prw.rewrite_op(op)

    return op
  }

}

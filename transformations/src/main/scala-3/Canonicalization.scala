package scair.transformations.canonicalization

import scair.dialects.arith.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.transformations.*
import scair.transformations.patterns.*

// TODO: Generalize in terms of effects and out of dialect
val RemoveUnusedOperations = pattern {
  case _: IsTerminator => PatternAction.Abort
  case op: NoMemoryEffect if op.results.forall(_.uses.isEmpty) =>
    PatternAction.Erase
  case op: NoMemoryEffect => PatternAction.Abort
}

// TODO: Generalize in Commutative/ConstantLike
val Commute = pattern { case c: Commutative =>
  val (const, nconst) = c.operands.partition(_.owner match
    case Some(_: ConstantLike) => true
    case _                     => false)
  val nops = nconst ++ const
  if nops == c.operands then PatternAction.Abort
  else c.updated(operands = nops)
}

// addi(addi(x, c0), c1) -> addi(x, c0 + c1)
val AddIAddConstant = pattern {
  case AddI(
        Owner(AddI(x, Owner(Constant(c0: IntegerAttr, _)), _)),
        Owner(Constant(c1: IntegerAttr, _)),
        _
      ) =>
    val cv = Result(x.typ)
    Seq(Constant(c0 + c1, cv), AddI(x, cv, Result(x.typ)))
}

object Canonicalize extends ModulePass {
  override val name = "canonicalize"

  override def transform(op: Operation): Operation = {
    val prw = new PatternRewriteWalker(
      GreedyRewritePatternApplier(
        Seq(
          RemoveUnusedOperations,
          Commute,
          AddIAddConstant
        )
      )
    )
    prw.rewrite_op(op)

    return op
  }

}

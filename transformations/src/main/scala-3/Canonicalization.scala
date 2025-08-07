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

// addi(subi(x, c0), c1) -> addi(x, c1 - c0)
val AddISubConstantRHS = pattern {
  case AddI(
        Owner(SubI(x, Owner(Constant(c0: IntegerAttr, _)), _)),
        Owner(Constant(c1: IntegerAttr, _)),
        _
      ) =>
    val cv = Result(x.typ)
    Seq(Constant(c0 - c1, cv), AddI(x, cv, Result(x.typ)))
}

// addi(subi(c0, x), c1) -> subi(c0 + c1, x)
val AddISubConstantLHS = pattern {
  case AddI(
        Owner(SubI(Owner(Constant(c0: IntegerAttr, _)), x, _)),
        Owner(Constant(c1: IntegerAttr, _)),
        _
      ) =>
    val cv = Result(x.typ)
    Seq(Constant(c0 + c1, cv), SubI(cv, x, Result(x.typ)))
}

// subi(addi(x, c0), c1) -> addi(x, c0 - c1)
val SubIRHSAddConstant = pattern {
  case SubI(
        Owner(AddI(x, Owner(Constant(c0: IntegerAttr, _)), _)),
        Owner(Constant(c1: IntegerAttr, _)),
        _
      ) =>
    val cv = Result(x.typ)
    Seq(Constant(c0 - c1, cv), AddI(x, cv, Result(x.typ)))
}

// subi(c1, addi(x, c0)) -> subi(c1 - c0, x)
val SubILHSAddConstant = pattern {
  case SubI(
        Owner(Constant(c1: IntegerAttr, _)),
        Owner(AddI(x, Owner(Constant(c0: IntegerAttr, _)), _)),
        _
      ) =>
    val cv = Result(x.typ)
    Seq(Constant(c1 - c0, cv), SubI(cv, x, Result(x.typ)))
}

// // subi(subi(x, c0), c1) -> subi(x, c0 + c1)
val SubIRHSSubConstantRHS = pattern {
  case SubI(
        Owner(SubI(x, Owner(Constant(c0: IntegerAttr, _)), _)),
        Owner(Constant(c1: IntegerAttr, _)),
        _
      ) =>
    val cv = Result(x.typ)
    Seq(Constant(c0 + c1, cv), SubI(x, cv, Result(x.typ)))
}

// subi(subi(c0, x), c1) -> subi(c0 - c1, x)
val SubIRHSSubConstantLHS = pattern {
  case SubI(
        Owner(SubI(Owner(Constant(c0: IntegerAttr, _)), x, _)),
        Owner(Constant(c1: IntegerAttr, _)),
        _
      ) =>
    val cv = Result(x.typ)
    Seq(Constant(c0 - c1, cv), SubI(cv, x, Result(x.typ)))
}

// subi(c1, subi(x, c0)) -> subi(c0 + c1, x)
val SubILHSSubConstantRHS = pattern {
  case SubI(
        Owner(Constant(c1: IntegerAttr, _)),
        Owner(SubI(x, Owner(Constant(c0: IntegerAttr, _)), _)),
        _
      ) =>
    val cv = Result(x.typ)
    Seq(Constant(c0 + c1, cv), SubI(cv, x, Result(x.typ)))
}

// // subi(c1, subi(c0, x)) -> addi(x, c1 - c0)
val SubILHSSubConstantLHS = pattern {
  case SubI(
        Owner(Constant(c1: IntegerAttr, _)),
        Owner(SubI(Owner(Constant(c0: IntegerAttr, _)), x, _)),
        _
      ) =>
    val cv = Result(x.typ)
    Seq(Constant(c1 - c0, cv), AddI(x, cv, Result(x.typ)))
}

// subi(subi(a, b), a) -> subi(0, b)
val SubISubILHSRHSLHS = pattern {
  case SubI(Owner(SubI(a, b, _)), aa, _) if a eq aa =>
    val cv = Result(a.typ)
    Seq(
      Constant(IntegerAttr(IntData(0), a.typ), cv),
      SubI(cv, b, Result(b.typ))
    )
}

object Canonicalize extends ModulePass {
  override val name = "canonicalize"

  override def transform(op: Operation): Operation = {
    val prw = new PatternRewriteWalker(
      GreedyRewritePatternApplier(
        Seq(
          RemoveUnusedOperations,
          Commute,
          AddIAddConstant,
          AddISubConstantRHS,
          AddISubConstantLHS,
          SubIRHSAddConstant,
          SubILHSAddConstant,
          SubIRHSSubConstantRHS,
          SubIRHSSubConstantLHS,
          SubILHSSubConstantRHS,
          SubILHSSubConstantLHS,
          SubISubILHSRHSLHS
        )
      )
    )
    prw.rewrite_op(op)

    return op
  }

}

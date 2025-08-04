package scair.transformations.cdt

import scair.dialects.arith.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.transformations.*

import scala.PartialFunction

object Owner {
  def unapply(v: Value[Attribute]) = v.owner
}

val AddIAddConstant = pattern { 
  case AddI(
        lhs = Owner(Constant(lv: IntegerAttr, _)),
        rhs = Owner(Constant(rv: IntegerAttr, _))
      ) =>
    Constant(
      IntegerAttr(lv.value.value + rv.value.value, lv.typ),
      Result(lv.typ)
    )
}

def pattern(p: PartialFunction[Operation, Operation]) =
  new RewritePattern {
    override def match_and_rewrite(
        op: Operation,
        rewriter: PatternRewriter
    ): Unit =
      p.lift(op).map(rewriter.replace_op(op, _))
  }

object ArithCanonicalize extends ModulePass {
  override val name = "arith-canonicalize"

  override def transform(op: Operation): Operation = {
    val prw = new PatternRewriteWalker(
      GreedyRewritePatternApplier(
        Seq(
          AddIAddConstant
        )
      )
    )
    prw.rewrite_op(op)

    return op
  }

}

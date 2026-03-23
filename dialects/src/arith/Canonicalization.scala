package scair.dialects.arith.canonicalization

import scair.dialects.arith.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.transformations.*
import scair.transformations.CanonicalizationPatterns

// AddI folding patterns
val AddIFold = pattern {
  // addi(x, 0) -> x
  case AddI(lhs = x, rhs = Owner(Constant(IntegerAttr(IntData(0), _), _))) =>
    (Seq(), Seq(x))
  // addi(subi(a, b), b) -> a
  case AddI(
        lhs = Owner(SubI(lhs = a, rhs = b)),
        rhs = bb,
      ) if b eq bb =>
    (Seq(), Seq(a))
  // addi(b, subi(a, b)) -> a
  case AddI(
        lhs = b,
        rhs = Owner(SubI(lhs = a, rhs = bb)),
      ) if b eq bb =>
    (Seq(), Seq(a))
  // addi(c0, c1) -> c0 + c1
  case AddI(
        lhs = Owner(Constant(c0: IntegerAttr, _)),
        rhs = Owner(Constant(c1: IntegerAttr, _)),
      ) =>
    Constant(c0 + c1, Result(c0.typ))
}

// AddI canonicalization patterns

// addi(addi(x, c0), c1) -> addi(x, c0 + c1)
val AddIAddConstant = pattern {
  case AddI(
        lhs = Owner(AddI(lhs = x, rhs = Owner(Constant(c0: IntegerAttr, _)))),
        rhs = Owner(Constant(c1: IntegerAttr, _)),
      ) =>
    val cv = Result(x.typ)
    Seq(Constant(c0 + c1, cv), AddI(x, cv, Result(x.typ)))
}

// addi(subi(x, c0), c1) -> addi(x, c1 - c0)
val AddISubConstantRHS = pattern {
  case AddI(
        lhs = Owner(SubI(lhs = x, rhs = Owner(Constant(c0: IntegerAttr, _)))),
        rhs = Owner(Constant(c1: IntegerAttr, _)),
      ) =>
    val cv = Result(x.typ)
    Seq(Constant(c0 - c1, cv), AddI(x, cv, Result(x.typ)))
}

// addi(x, muli(y, -1)) -> subi(x, y)
val AddIMulNegativeOneRhs = pattern {
  case AddI(
        lhs = x,
        rhs = Owner(
          MulI(lhs = y, rhs = Owner(Constant(IntegerAttr(IntData(-1), _), _)))
        ),
      ) =>
    Seq(SubI(x, y, Result(x.typ)))
}

// addi(muli(x, -1), y) -> subi(y, x)
val AddIMulNegativeOneLhs = pattern {
  case AddI(
        lhs = Owner(
          MulI(lhs = x, rhs = Owner(Constant(IntegerAttr(IntData(-1), _), _)))
        ),
        rhs = y,
      ) =>
    Seq(SubI(y, x, Result(x.typ)))
}

// addi(subi(c0, x), c1) -> subi(c0 + c1, x)
val AddISubConstantLHS = pattern {
  case AddI(
        lhs = Owner(SubI(lhs = Owner(Constant(c0: IntegerAttr, _)), rhs = x)),
        rhs = Owner(Constant(c1: IntegerAttr, _)),
      ) =>
    val cv = Result(x.typ)
    Seq(Constant(c0 + c1, cv), SubI(cv, x, Result(x.typ)))
}

given [T <: AnyIntegerType]
    => CanonicalizationPatterns[AddI[T]](
      AddIFold,
      AddIAddConstant,
      AddISubConstantRHS,
      AddIMulNegativeOneRhs,
      AddIMulNegativeOneLhs,
      AddISubConstantLHS,
    )

// muli(muli(x, c0), c1) -> muli(x, c0 * c1)
val MulIMulIConstant = pattern {
  case MulI(
        lhs = Owner(MulI(lhs = x, rhs = Owner(Constant(c0: IntegerAttr, _)))),
        rhs = Owner(Constant(c1: IntegerAttr, _)),
      ) =>
    val cv = Result(x.typ)
    Seq(Constant(c0 * c1, cv), MulI(x, cv, Result(x.typ)))
}

// MulI folding patterns
val MulIFold = pattern {
  // muli(x, 0) -> 0
  case MulI(lhs = x, rhs = Owner(Constant(IntegerAttr(IntData(0), _), _))) =>
    Constant(IntegerAttr(IntData(0), x.typ), Result(x.typ))
  // muli(x, 1) -> x
  case MulI(lhs = x, rhs = Owner(Constant(IntegerAttr(IntData(1), _), _))) =>
    (Seq(), Seq(x))
  // muli(c0, c1) -> c0 * c1
  case MulI(
        lhs = Owner(Constant(c0: IntegerAttr, _)),
        rhs = Owner(Constant(c1: IntegerAttr, _)),
      ) =>
    Constant(c0 * c1, Result(c0.typ))
}

given CanonicalizationPatterns[MulI](
  MulIMulIConstant,
  MulIFold,
)

// SubI folding patterns
val SubIFold = pattern {
  // subi(x,x) -> 0
  case SubI(lhs = x, rhs = y) if x eq y =>
    Constant(IntegerAttr(IntData(0), x.typ), Result(x.typ))
  // subi(x,0) -> x
  case SubI(lhs = x, rhs = Owner(Constant(IntegerAttr(IntData(0), _), _))) =>
    (Seq(), Seq(x))
  // subi(addi(a, b), b) -> a
  case SubI(
        lhs = Owner(AddI(lhs = a, rhs = b)),
        rhs = bb,
      ) if b eq bb =>
    (Seq(), Seq(a))
  // subi(addi(a, b), a) -> b
  case SubI(
        lhs = Owner(AddI(lhs = a, rhs = b)),
        rhs = aa,
      ) if a eq aa =>
    (Seq(), Seq(b))
  // subi(c0, c1) -> c0 - c1
  case SubI(
        lhs = Owner(Constant(c0: IntegerAttr, _)),
        rhs = Owner(Constant(c1: IntegerAttr, _)),
      ) =>
    Constant(c0 - c1, Result(c0.typ))
}

// SubI canonicalization patterns
// subi(addi(x, c0), c1) -> addi(x, c0 - c1)
val SubIRHSAddConstant = pattern {
  case SubI(
        lhs = Owner(AddI(lhs = x, rhs = Owner(Constant(c0: IntegerAttr, _)))),
        rhs = Owner(Constant(c1: IntegerAttr, _)),
      ) =>
    val cv = Result(x.typ)
    Seq(Constant(c0 - c1, cv), AddI(x, cv, Result(x.typ)))
}

// subi(c1, addi(x, c0)) -> subi(c1 - c0, x)
val SubILHSAddConstant = pattern {
  case SubI(
        lhs = Owner(Constant(c1: IntegerAttr, _)),
        rhs = Owner(AddI(lhs = x, rhs = Owner(Constant(c0: IntegerAttr, _)))),
      ) =>
    val cv = Result(x.typ)
    Seq(Constant(c1 - c0, cv), SubI(cv, x, Result(x.typ)))
}

// // subi(subi(x, c0), c1) -> subi(x, c0 + c1)
val SubIRHSSubConstantRHS = pattern {
  case SubI(
        lhs = Owner(SubI(lhs = x, rhs = Owner(Constant(c0: IntegerAttr, _)))),
        rhs = Owner(Constant(c1: IntegerAttr, _)),
      ) =>
    val cv = Result(x.typ)
    Seq(Constant(c0 + c1, cv), SubI(x, cv, Result(x.typ)))
}

// subi(subi(c0, x), c1) -> subi(c0 - c1, x)
val SubIRHSSubConstantLHS = pattern {
  case SubI(
        lhs = Owner(SubI(lhs = Owner(Constant(c0: IntegerAttr, _)), rhs = x)),
        rhs = Owner(Constant(c1: IntegerAttr, _)),
      ) =>
    val cv = Result(x.typ)
    Seq(Constant(c0 - c1, cv), SubI(cv, x, Result(x.typ)))
}

// subi(c1, subi(x, c0)) -> subi(c0 + c1, x)
val SubILHSSubConstantRHS = pattern {
  case SubI(
        lhs = Owner(Constant(c1: IntegerAttr, _)),
        rhs = Owner(SubI(lhs = x, rhs = Owner(Constant(c0: IntegerAttr, _)))),
      ) =>
    val cv = Result(x.typ)
    Seq(Constant(c0 + c1, cv), SubI(cv, x, Result(x.typ)))
}

// // subi(c1, subi(c0, x)) -> addi(x, c1 - c0)
val SubILHSSubConstantLHS = pattern {
  case SubI(
        lhs = Owner(Constant(c1: IntegerAttr, _)),
        rhs = Owner(SubI(lhs = Owner(Constant(c0: IntegerAttr, _)), rhs = x)),
      ) =>
    val cv = Result(x.typ)
    Seq(Constant(c1 - c0, cv), AddI(x, cv, Result(x.typ)))
}

// subi(subi(a, b), a) -> subi(0, b)
val SubISubILHSRHSLHS = pattern {
  case SubI(
        lhs = Owner(SubI(lhs = a, rhs = b)),
        rhs = aa,
      ) if a eq aa =>
    val cv = Result(a.typ)
    Seq(
      Constant(IntegerAttr(IntData(0), a.typ), cv),
      SubI(cv, b, Result(b.typ)),
    )
}

given CanonicalizationPatterns[SubI](
  SubIFold,
  SubIRHSAddConstant,
  SubILHSAddConstant,
  SubIRHSSubConstantRHS,
  SubIRHSSubConstantLHS,
  SubILHSSubConstantRHS,
  SubILHSSubConstantLHS,
  SubISubILHSRHSLHS,
)

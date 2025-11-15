package scair.dialects.complex.canonicalization

import scair.dialects.arith
import scair.dialects.builtin.*
import scair.dialects.complex.*
import scair.ir.*
import scair.transformations.CanonicalizationPatterns
import scair.transformations.patterns.*

// ░█████╗░ ░█████╗░ ███╗░░░███╗ ██████╗░ ██╗░░░░░ ███████╗ ██╗░░██╗
// ██╔══██╗ ██╔══██╗ ████╗░████║ ██╔══██╗ ██║░░░░░ ██╔════╝ ╚██╗██╔╝
// ██║░░╚═╝ ██║░░██║ ██╔████╔██║ ██████╔╝ ██║░░░░░ █████╗░░ ░╚███╔╝░
// ██║░░██╗ ██║░░██║ ██║╚██╔╝██║ ██╔═══╝░ ██║░░░░░ ██╔══╝░░ ░██╔██╗░
// ╚█████╔╝ ╚█████╔╝ ██║░╚═╝░██║ ██║░░░░░ ███████╗ ███████╗ ██╔╝╚██╗
// ░╚════╝░ ░╚════╝░ ╚═╝░░░░░╚═╝ ╚═╝░░░░░ ╚══════╝ ╚══════╝ ╚═╝░░╚═╝
//
// ░█████╗░ ░█████╗░ ███╗░░██╗ ░█████╗░ ███╗░░██╗ ██╗ ░█████╗░ ░█████╗░ ██╗░░░░░ ██╗ ███████╗ ░█████╗░ ████████╗ ██╗ ░█████╗░ ███╗░░██╗
// ██╔══██╗ ██╔══██╗ ████╗░██║ ██╔══██╗ ████╗░██║ ██║ ██╔══██╗ ██╔══██╗ ██║░░░░░ ██║ ╚════██║ ██╔══██╗ ╚══██╔══╝ ██║ ██╔══██╗ ████╗░██║
// ██║░░╚═╝ ███████║ ██╔██╗██║ ██║░░██║ ██╔██╗██║ ██║ ██║░░╚═╝ ███████║ ██║░░░░░ ██║ ░░███╔═╝ ███████║ ░░░██║░░░ ██║ ██║░░██║ ██╔██╗██║
// ██║░░██╗ ██╔══██║ ██║╚████║ ██║░░██║ ██║╚████║ ██║ ██║░░██╗ ██╔══██║ ██║░░░░░ ██║ ██╔══╝░░ ██╔══██║ ░░░██║░░░ ██║ ██║░░██║ ██║╚████║
// ╚█████╔╝ ██║░░██║ ██║░╚███║ ╚█████╔╝ ██║░╚███║ ██║ ╚█████╔╝ ██║░░██║ ███████╗ ██║ ███████╗ ██║░░██║ ░░░██║░░░ ██║ ╚█████╔╝ ██║░╚███║
// ░╚════╝░ ╚═╝░░╚═╝ ╚═╝░░╚══╝ ░╚════╝░ ╚═╝░░╚══╝ ╚═╝ ░╚════╝░ ╚═╝░░╚═╝ ╚══════╝ ╚═╝ ╚══════╝ ╚═╝░░╚═╝ ░░░╚═╝░░░ ╚═╝ ░╚════╝░ ╚═╝░░╚══╝

// complex.create(complex.re(op), complex.im(op)) -> op
val CreateReIm = pattern {
  case Create(Owner(Re(op, _)), Owner(Im(oq, _)), _) if op eq oq =>
    (Seq(), Seq(op))
}

given CanonicalizationPatterns[Create](
  CreateReIm
)

// complex.re(complex.constant(a, b)) -> a
val ReConstant = pattern {
  case Re(Owner(Constant(ArrayAttribute(Seq(r, _)), _)), res) =>
    arith.Constant(r, res.copy())
}

// complex.re(complex.create(a, b)) -> a
val ReCreate = pattern { case Re(Owner(Create(a, b, _)), _) =>
  (Seq(), Seq(a))
}

// complex.re(complex.neg(complex.create(a, b))) -> -a
val ReNegCreate = pattern {
  case Re(
        Owner(Neg(Owner(Create(a @ Value(at: FloatType), _, _)), _, fastmath)),
        _
      ) =>
    arith.NegF(a.asInstanceOf[Value[FloatType]], Result(at), fastmath)
}

given CanonicalizationPatterns[Re](
  ReConstant,
  ReCreate,
  ReNegCreate
)

// complex.im(complex.constant(a, b)) -> b
val ImConstant = pattern {
  case Im(Owner(Constant(ArrayAttribute(Seq(_, i)), _)), res) =>
    arith.Constant(i, res.copy())
}

// complex.im(complex.create(a, b)) -> b
val ImCreate = pattern { case Im(Owner(Create(a, b, _)), _) =>
  (Seq(), Seq(b))
}

// complex.im(complex.neg(complex.create(a, b))) -> -b
val ImNegCreate = pattern {
  case Im(
        Owner(Neg(Owner(Create(_, b @ Value(bt: FloatType), _)), _, fastmath)),
        _
      ) =>
    arith.NegF(b.asInstanceOf[Value[FloatType]], Result(bt), fastmath)
}

given CanonicalizationPatterns[Im](
  ImConstant,
  ImCreate,
  ImNegCreate
)

// complex.add(complex.sub(a, b), b) -> a
val AddSub = pattern {
  case Add(
        Owner(Sub(a, b, _, _)),
        bb,
        _,
        _
      ) if b eq bb =>
    (Seq(), Seq(a))
}

// complex.add(b, complex.sub(a, b)) -> a
val AddSubRHS = pattern {
  case Add(
        b,
        Owner(Sub(a, bb, _, _)),
        _,
        _
      ) if b eq bb =>
    (Seq(), Seq(a))
}

// complex.add(a, complex.constant<0.0, 0.0>) -> a
val AddZero = pattern {
  case Add(
        a,
        Owner(
          Constant(
            ArrayAttribute(
              Seq(FloatAttr(FloatData(r), _), FloatAttr(FloatData(i), _))
            ),
            _
          )
        ),
        _,
        _
      ) if r == 0.0 && i == 0.0 =>
    (Seq(), Seq(a))
}

given CanonicalizationPatterns[Add](
  AddSub,
  AddSubRHS,
  AddZero
)

// complex.sub(complex.add(a, b), b) -> a
val SubAdd = pattern {
  case Sub(
        Owner(Add(a, b, _, _)),
        bb,
        _,
        _
      ) if b eq bb =>
    (Seq(), Seq(a))
}

// complex.sub(a, complex.constant<0.0, 0.0>) -> a
val SubZero = pattern {
  case Sub(
        a,
        Owner(
          Constant(
            ArrayAttribute(
              Seq(FloatAttr(FloatData(r), _), FloatAttr(FloatData(i), _))
            ),
            _
          )
        ),
        _,
        _
      ) if r == 0.0 && i == 0.0 =>
    (Seq(), Seq(a))
}

given CanonicalizationPatterns[Sub](
  SubAdd,
  SubZero
)

// complex.neg(complex.neg(a)) -> a
val NegNeg = pattern { case Neg(Owner(Neg(a, _, _)), _, _) =>
  (Seq(), Seq(a))
}

given CanonicalizationPatterns[Neg](
  NegNeg
)

// complex.mul(a, complex.constant<1.0, 0.0>) -> a
val MulOne = pattern {
  case Mul(
        a,
        Owner(
          Constant(
            ArrayAttribute(
              Seq(FloatAttr(FloatData(r), _), FloatAttr(FloatData(i), _))
            ),
            _
          )
        ),
        _,
        _
      ) if r == 1.0 && i == 0.0 =>
    (Seq(), Seq(a))
}

// complex.mul(a, complex.constant<0.0, 0.0>) -> complex.constant<0.0, 0.0>
val MulZero = pattern {
  case Mul(
        a,
        zero @ Owner(
          Constant(
            ArrayAttribute(
              Seq(FloatAttr(FloatData(r), _), FloatAttr(FloatData(i), _))
            ),
            _
          )
        ),
        _,
        _
      ) if r == 0.0 && i == 0.0 =>
    (Seq(), Seq(zero))
}

// (a + bi) * (c + di) = (ac-bd) + (ad+bc)i
val MulConstant = pattern {
  case Mul(
        Owner(
          Constant(
            ArrayAttribute(
              Seq(FloatAttr(FloatData(a), t), FloatAttr(FloatData(b), _))
            ),
            _
          )
        ),
        Owner(
          Constant(
            ArrayAttribute(
              Seq(FloatAttr(FloatData(c), _), FloatAttr(FloatData(d), _))
            ),
            _
          )
        ),
        Result(tpe),
        _
      ) =>
    Constant(
      ArrayAttribute(
        Seq(
          FloatAttr(FloatData(a * c - b * d), t),
          FloatAttr((FloatData(a * d + b * c)), t)
        )
      ),
      Result(tpe)
    )
}

given CanonicalizationPatterns[Mul](
  MulOne,
  MulZero,
  MulConstant
)

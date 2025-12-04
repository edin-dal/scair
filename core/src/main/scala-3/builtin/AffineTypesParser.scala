package scair.dialects.affine

import fastparse.*
import scair.AttrParser.whitespace
import scair.Parser.*

// ░█████╗░ ███████╗ ███████╗ ██╗ ███╗░░██╗ ███████╗
// ██╔══██╗ ██╔════╝ ██╔════╝ ██║ ████╗░██║ ██╔════╝
// ███████║ █████╗░░ █████╗░░ ██║ ██╔██╗██║ █████╗░░
// ██╔══██║ ██╔══╝░░ ██╔══╝░░ ██║ ██║╚████║ ██╔══╝░░
// ██║░░██║ ██║░░░░░ ██║░░░░░ ██║ ██║░╚███║ ███████╗
// ╚═╝░░╚═╝ ╚═╝░░░░░ ╚═╝░░░░░ ╚═╝ ╚═╝░░╚══╝ ╚══════╝

// ████████╗ ██╗░░░██╗ ██████╗░ ███████╗ ░██████╗
// ╚══██╔══╝ ╚██╗░██╔╝ ██╔══██╗ ██╔════╝ ██╔════╝
// ░░░██║░░░ ░╚████╔╝░ ██████╔╝ █████╗░░ ╚█████╗░
// ░░░██║░░░ ░░╚██╔╝░░ ██╔═══╝░ ██╔══╝░░ ░╚═══██╗
// ░░░██║░░░ ░░░██║░░░ ██║░░░░░ ███████╗ ██████╔╝
// ░░░╚═╝░░░ ░░░╚═╝░░░ ╚═╝░░░░░ ╚══════╝ ╚═════╝░

// ██████╗░ ░█████╗░ ██████╗░ ░██████╗ ███████╗ ██████╗░
// ██╔══██╗ ██╔══██╗ ██╔══██╗ ██╔════╝ ██╔════╝ ██╔══██╗
// ██████╔╝ ███████║ ██████╔╝ ╚█████╗░ █████╗░░ ██████╔╝
// ██╔═══╝░ ██╔══██║ ██╔══██╗ ░╚═══██╗ ██╔══╝░░ ██╔══██╗
// ██║░░░░░ ██║░░██║ ██║░░██║ ██████╔╝ ███████╗ ██║░░██║
// ╚═╝░░░░░ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ╚═════╝░ ╚══════╝ ╚═╝░░╚═╝

/*≡≡=---=≡≡≡=---=≡≡*\
||      UTILS      ||
\*≡==----=≡=----==≡*/

def checkDistinct[$: P](name: String, list: Seq[String]): P[Seq[String]] =
  if list.distinct.size != list.size then
    Fail(
      s"Number of ${name} in Affine Map/Set must be unique! ;)"
    )
  else Pass(list)

def validateAffineExpr[$: P](
    name: String,
    dimsym: String,
    list: Seq[String]
): P[String] =
  if !list.contains(dimsym) then
    println(list.contains(dimsym))
    println(list)
    println(dimsym)
    Fail(
      s"${name} \"${dimsym}\" used in the expression but not defined! | ${dimsym} | ${list}"
    )
  else Pass(dimsym)

val add = AffineBinaryOp.Add
val minus = AffineBinaryOp.Minus
val multiply = AffineBinaryOp.Multiply
val ceildiv = AffineBinaryOp.CeilDiv
val floordiv = AffineBinaryOp.FloorDiv
val mod = AffineBinaryOp.Mod

val greaterequal = AffineConstraintKind.GreaterEqual
val lessequal = AffineConstraintKind.LessEqual
val equal = AffineConstraintKind.Equal

/*≡≡=---==≡≡≡==---=≡≡*\
||    DIMS & SYMS    ||
\*≡==----==≡==----==≡*/

// dim-use-list            ::= `(` ssa-use-list? `)`
// symbol-use-list         ::= `[` ssa-use-list? `]`
// dim-and-symbol-use-list ::= dim-use-list symbol-use-list?

def DimUseP[$: P]: P[String] = P("d" ~~ DecimalLiteral).!

def SymUseP[$: P]: P[String] = P("s" ~~ DecimalLiteral).!

def DimUseListP[$: P]: P[Seq[String]] =
  P("(" ~ DimUseP.rep(0, sep = ",") ~ ")")
    .flatMap(checkDistinct("dimensions", _))

def SymUseListP[$: P]: P[Seq[String]] =
  P("[" ~ SymUseP.rep(0, sep = ",") ~ "]").flatMap(checkDistinct("symbols", _))

def DimSymUseListP[$: P]: P[(Seq[String], Seq[String])] =
  P(DimUseListP ~ SymUseListP.rep(min = 0, max = 1).map(_.flatten))

/*≡==---==≡≡≡==---=≡≡*\
||    AFFINE EXPR    ||
\*≡==----==≡==----==≡*/

//  affine-expr ::= affine-singles `+` affine-expr
//                | affine-singles `-` affine-expr
//                | `-`? integer-literal `*` affine-expr
//                | affine-singles `ceildiv` integer-literal
//                | affine-singles `floordiv` integer-literal
//                | affine-singles `mod` integer-literal

//  multi-dim-affine-expr ::= `(` `)`
//                          | `(` affine-expr (`,` affine-expr)* `)`

//  affine-singles        ::= `(` affine-expr `)`
//                         | `-`affine-singles
//                         | bare-id
//                         | integer-literal

def AffineSinglesP[$: P](dims: Seq[String], symbs: Seq[String]): P[AffineExpr] =
  P(
    "(" ~ AffineExprP(dims, symbs) ~ ")" |
      "-" ~ AffineSinglesP(dims, symbs) |
      AffineDimExprP(dims, symbs) |
      AffineSymExprP(dims, symbs) |
      AffineConstantP(dims, symbs)
  )

def AffineDimExprP[$: P](dims: Seq[String], symbs: Seq[String]): P[AffineExpr] =
  P(DimUseP)
    .flatMap(validateAffineExpr("dimension", _, dims))
    .map(AffineDimExpr(_))

def AffineSymExprP[$: P](dims: Seq[String], symbs: Seq[String]): P[AffineExpr] =
  P(SymUseP)
    .flatMap(validateAffineExpr("symbol", _, symbs))
    .map(AffineSymExpr(_))

def AffineConstantP[$: P](
    dims: Seq[String],
    symbs: Seq[String]
): P[AffineExpr] =
  P(IntegerLiteral).map(AffineConstantExpr(_))

def AffineExprP[$: P](dims: Seq[String], symbs: Seq[String]): P[AffineExpr] =
  P(
    AffineAddExpr(dims, symbs) |
      AffineMinusExpr(dims, symbs) |
      AffineMultiplyExpr(dims, symbs) |
      AffineCeilDivExpr(dims, symbs) |
      AffineFloorDivExpr(dims, symbs) |
      AffineModExpr(dims, symbs) |
      AffineSinglesP(dims, symbs)
  )

def AffineAddExpr[$: P](
    dims: Seq[String],
    symbs: Seq[String]
): P[AffineBinaryOpExpr] =
  P(AffineSinglesP(dims, symbs) ~ "+" ~ AffineExprP(dims, symbs))
    .map(AffineBinaryOpExpr(add, _, _))

def AffineMinusExpr[$: P](
    dims: Seq[String],
    symbs: Seq[String]
): P[AffineBinaryOpExpr] =
  P(AffineSinglesP(dims, symbs) ~ "-" ~ AffineExprP(dims, symbs))
    .map(AffineBinaryOpExpr(minus, _, _))

def AffineMultiplyExpr[$: P](
    dims: Seq[String],
    symbs: Seq[String]
): P[AffineBinaryOpExpr] =
  P(
    AffineConstantP(dims, symbs) ~ "*" ~ AffineSinglesP(
      dims,
      symbs
    ) | AffineSinglesP(dims, symbs) ~ "*" ~ AffineConstantP(dims, symbs)
  )
    .map(AffineBinaryOpExpr(multiply, _, _))

def AffineCeilDivExpr[$: P](
    dims: Seq[String],
    symbs: Seq[String]
): P[AffineBinaryOpExpr] =
  P(AffineSinglesP(dims, symbs) ~ "ceildiv" ~ AffineConstantP(dims, symbs))
    .map(AffineBinaryOpExpr(ceildiv, _, _))

def AffineFloorDivExpr[$: P](
    dims: Seq[String],
    symbs: Seq[String]
): P[AffineBinaryOpExpr] =
  P(AffineSinglesP(dims, symbs) ~ "floordiv" ~ AffineConstantP(dims, symbs))
    .map(AffineBinaryOpExpr(floordiv, _, _))

def AffineModExpr[$: P](
    dims: Seq[String],
    symbs: Seq[String]
): P[AffineBinaryOpExpr] =
  P(AffineSinglesP(dims, symbs) ~ "mod" ~ AffineConstantP(dims, symbs))
    .map(AffineBinaryOpExpr(mod, _, _))

def MultiDimAffineExpr[$: P](
    dims: Seq[String],
    symbs: Seq[String]
): P[Seq[AffineExpr]] =
  P("(" ~ AffineExprP(dims, symbs).rep(0, sep = ",") ~ ")")

/*≡==---==≡≡≡≡==---=≡≡*\
||     AFFINE MAP     ||
\*≡==----==≡≡==----==≡*/

//  affine-map-inline  ::= dim-and-symbol-value-lists `->` multi-dim-affine-expr
//  affine-map-id      ::= `#` suffix-id
//  affine-map-def     ::= affine-map-id `=` affine-map-inline
//  module-header-def  ::= affine-map-def
//  affine-map         ::= affine-map-id | affine-map-inline

def AffineMapP[$: P]: P[AffineMap] =
  P(DimSymUseListP.flatMap { (x: (Seq[String], Seq[String])) =>
    P("->" ~ MultiDimAffineExpr(x._1, x._2)).map(AffineMap(x._1, x._2, _))
  })

// TODO : implement logic for these two
// def AffineMapID[$: P]: P[Any] = P( "#" ~ SuffixId )
// def AffineMapDef[$: P]: P[Any] = P( AffineMapID ~ "=" ~ AffineMapInline)
// def AffineMap[$: P]: P[Any] = P( AffineMapInline )

/*≡==---==≡≡≡≡==---=≡≡*\
||     AFFINE SET     ||
\*≡==----==≡≡==----==≡*/

//  integer-set-id      ::= `#` suffix-id
//  integer-set-inline  ::= dim-and-symbol-value-lists `:` '(' affine-constraint-conjunction? ')'
//  integer-set-decl    ::= integer-set-id `=` integer-set-inline
//  integer-set         ::= integer-set-id | integer-set-inline
//  affine-constraint   ::= affine-expr `>=` `affine-expr`
//                        | affine-expr `<=` `affine-expr`
//                        | affine-expr `==` `affine-expr`
//  affine-constraint-conjunction
//    ::= affine-constraint (`,` affine-constraint)*

def AffineSetP[$: P]: P[AffineSet] =
  P(DimSymUseListP.flatMap { (x: (Seq[String], Seq[String])) =>
    P(":" ~ "(" ~ AffineConstraintP(x._1, x._2).rep(0, sep = ",") ~ ")")
      .map(AffineSet(x._1, x._2, _))
  })

def AffineConstraintP[$: P](
    dims: Seq[String],
    symbs: Seq[String]
): P[AffineConstraintExpr] =
  P(GreaterEqualP(dims, symbs) | LessEqualP(dims, symbs) | EqualP(dims, symbs))

def GreaterEqualP[$: P](
    dims: Seq[String],
    symbs: Seq[String]
): P[AffineConstraintExpr] =
  P(AffineExprP(dims, symbs) ~ ">=" ~ AffineExprP(dims, symbs)).map(
    AffineConstraintExpr(greaterequal, _, _)
  )

def LessEqualP[$: P](
    dims: Seq[String],
    symbs: Seq[String]
): P[AffineConstraintExpr] =
  P(AffineExprP(dims, symbs) ~ "<=" ~ AffineExprP(dims, symbs)).map(
    AffineConstraintExpr(lessequal, _, _)
  )

def EqualP[$: P](
    dims: Seq[String],
    symbs: Seq[String]
): P[AffineConstraintExpr] =
  P(AffineExprP(dims, symbs) ~ "==" ~ AffineExprP(dims, symbs)).map(
    AffineConstraintExpr(equal, _, _)
  )

object TestAffine:

  def main(args: Array[String]): Unit =
    val parsed =
      parse("d0 + d1", AffineExprP(Seq("d0", "d1"), Seq())(using _))
    println(parsed)
    val parsed1 =
      parse("(d0, d1)[s0] -> (d0 + d1 + s0, d1 + s0)", AffineMapP(using _))
    println(parsed1)
    val parsed2 =
      parse("(d0, d1)[s0] : (d0 + d1 + s0 >= d1 + s0)", AffineSetP(using _))
    println(parsed2)

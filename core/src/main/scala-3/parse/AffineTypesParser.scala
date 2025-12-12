package scair.parse

import fastparse.*
import scair.*
import scair.dialects.affine.*

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

def checkDistinctP[$: P](name: String, list: Seq[String]): P[Seq[String]] =
  if list.distinct.size != list.size then
    Fail(
      s"Number of $name in Affine Map/Set must be unique! ;)"
    )
  else Pass(list)

def validateAffineExprP[$: P](
    name: String,
    dimsym: String,
    list: Seq[String],
): P[String] =
  if !list.contains(dimsym) then
    println(list.contains(dimsym))
    println(list)
    println(dimsym)
    Fail(
      s"$name \"$dimsym\" used in the expression but not defined! | $dimsym | $list"
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

def dimUseP[$: P]: P[String] = P("d" ~~ decimalLiteralP).!

def symUseP[$: P]: P[String] = P("s" ~~ decimalLiteralP).!

def dimUseListP[$: P]: P[Seq[String]] =
  P("(" ~ dimUseP.rep(0, sep = ",") ~ ")")
    .flatMap(checkDistinctP("dimensions", _))

def symUseListP[$: P]: P[Seq[String]] =
  P("[" ~ symUseP.rep(0, sep = ",") ~ "]").flatMap(checkDistinctP("symbols", _))

def dimSymUseListP[$: P]: P[(Seq[String], Seq[String])] =
  P(dimUseListP ~ symUseListP.rep(min = 0, max = 1).map(_.flatten))

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

def affineSinglesP[$: P](dims: Seq[String], symbs: Seq[String]): P[AffineExpr] =
  P(
    "(" ~ affineExprP(dims, symbs) ~ ")" | "-" ~ affineSinglesP(dims, symbs) |
      affineDimExprP(dims, symbs) | affineSymExprP(dims, symbs) |
      affineConstantP(dims, symbs)
  )

def affineDimExprP[$: P](
    dims: Seq[String],
    symbs: Seq[String],
): P[AffineDimExpr] =
  P(dimUseP).flatMap(validateAffineExprP("dimension", _, dims))
    .map(AffineDimExpr(_))

def affineSymExprP[$: P](
    dims: Seq[String],
    symbs: Seq[String],
): P[AffineSymExpr] =
  P(symUseP).flatMap(validateAffineExprP("symbol", _, symbs))
    .map(AffineSymExpr(_))

def affineConstantP[$: P](
    dims: Seq[String],
    symbs: Seq[String],
): P[AffineExpr] =
  P(integerLiteralP).map(AffineConstantExpr(_))

def affineExprP[$: P](dims: Seq[String], symbs: Seq[String]): P[AffineExpr] =
  P(
    affineAddExprP(dims, symbs) | affineMinusExprP(dims, symbs) |
      affineMultiplyExprP(dims, symbs) | affineCeilDivExprP(dims, symbs) |
      affineFloorDivExprP(dims, symbs) | affineModExprP(dims, symbs) |
      affineSinglesP(dims, symbs)
  )

def affineAddExprP[$: P](
    dims: Seq[String],
    symbs: Seq[String],
): P[AffineBinaryOpExpr] =
  P(affineSinglesP(dims, symbs) ~ "+" ~ affineExprP(dims, symbs))
    .map(AffineBinaryOpExpr(add, _, _))

def affineMinusExprP[$: P](
    dims: Seq[String],
    symbs: Seq[String],
): P[AffineBinaryOpExpr] =
  P(affineSinglesP(dims, symbs) ~ "-" ~ affineExprP(dims, symbs))
    .map(AffineBinaryOpExpr(minus, _, _))

def affineMultiplyExprP[$: P](
    dims: Seq[String],
    symbs: Seq[String],
): P[AffineBinaryOpExpr] =
  P(
    affineConstantP(dims, symbs) ~ "*" ~ affineSinglesP(
      dims,
      symbs,
    ) | affineSinglesP(dims, symbs) ~ "*" ~ affineConstantP(dims, symbs)
  ).map(AffineBinaryOpExpr(multiply, _, _))

def affineCeilDivExprP[$: P](
    dims: Seq[String],
    symbs: Seq[String],
): P[AffineBinaryOpExpr] =
  P(affineSinglesP(dims, symbs) ~ "ceildiv" ~ affineConstantP(dims, symbs))
    .map(AffineBinaryOpExpr(ceildiv, _, _))

def affineFloorDivExprP[$: P](
    dims: Seq[String],
    symbs: Seq[String],
): P[AffineBinaryOpExpr] =
  P(affineSinglesP(dims, symbs) ~ "floordiv" ~ affineConstantP(dims, symbs))
    .map(AffineBinaryOpExpr(floordiv, _, _))

def affineModExprP[$: P](
    dims: Seq[String],
    symbs: Seq[String],
): P[AffineBinaryOpExpr] =
  P(affineSinglesP(dims, symbs) ~ "mod" ~ affineConstantP(dims, symbs))
    .map(AffineBinaryOpExpr(mod, _, _))

def multiDimAffineExprP[$: P](
    dims: Seq[String],
    symbs: Seq[String],
): P[Seq[AffineExpr]] =
  P("(" ~ affineExprP(dims, symbs).rep(0, sep = ",") ~ ")")

/*≡==---==≡≡≡≡==---=≡≡*\
||     AFFINE MAP     ||
\*≡==----==≡≡==----==≡*/

//  affine-map-inline  ::= dim-and-symbol-value-lists `->` multi-dim-affine-expr
//  affine-map-id      ::= `#` suffix-id
//  affine-map-def     ::= affine-map-id `=` affine-map-inline
//  module-header-def  ::= affine-map-def
//  affine-map         ::= affine-map-id | affine-map-inline

def affineMapP[$: P]: P[AffineMap] =
  P(dimSymUseListP.flatMap { (x: (Seq[String], Seq[String])) =>
    P("->" ~ multiDimAffineExprP(x._1, x._2)).map(AffineMap(x._1, x._2, _))
  })

// TODO : implement logic for these two
// def affineMapIDP[$: P]: P[Any] = P( "#" ~ SuffixId )
// def affineMapDefP[$: P]: P[Any] = P( AffineMapID ~ "=" ~ AffineMapInline)
// def affineMapP[$: P]: P[Any] = P( AffineMapInline )

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

def affineSetP[$: P]: P[AffineSet] =
  P(dimSymUseListP.flatMap { (x: (Seq[String], Seq[String])) =>
    P(":" ~ "(" ~ affineConstraintP(x._1, x._2).rep(0, sep = ",") ~ ")")
      .map(AffineSet(x._1, x._2, _))
  })

def affineConstraintP[$: P](
    dims: Seq[String],
    symbs: Seq[String],
): P[AffineConstraintExpr] =
  P(greaterEqualP(dims, symbs) | lessEqualP(dims, symbs) | equalP(dims, symbs))

def greaterEqualP[$: P](
    dims: Seq[String],
    symbs: Seq[String],
): P[AffineConstraintExpr] =
  P(affineExprP(dims, symbs) ~ ">=" ~ affineExprP(dims, symbs))
    .map(
      AffineConstraintExpr(greaterequal, _, _)
    )

def lessEqualP[$: P](
    dims: Seq[String],
    symbs: Seq[String],
): P[AffineConstraintExpr] =
  P(affineExprP(dims, symbs) ~ "<=" ~ affineExprP(dims, symbs))
    .map(
      AffineConstraintExpr(lessequal, _, _)
    )

def equalP[$: P](
    dims: Seq[String],
    symbs: Seq[String],
): P[AffineConstraintExpr] =
  P(affineExprP(dims, symbs) ~ "==" ~ affineExprP(dims, symbs))
    .map(
      AffineConstraintExpr(equal, _, _)
    )

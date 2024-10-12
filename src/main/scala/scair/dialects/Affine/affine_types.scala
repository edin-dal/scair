package scair.dialects.affine

import scair.{
  ListType,
  DictType,
  Attribute,
  TypeAttribute,
  ParametrizedAttribute,
  DataAttribute
}

// ██████╗░ ░█████╗░ ░██████╗ ██╗ ░█████╗░
// ██╔══██╗ ██╔══██╗ ██╔════╝ ██║ ██╔══██╗
// ██████╦╝ ███████║ ╚█████╗░ ██║ ██║░░╚═╝
// ██╔══██╗ ██╔══██║ ░╚═══██╗ ██║ ██║░░██╗
// ██████╦╝ ██║░░██║ ██████╔╝ ██║ ╚█████╔╝
// ╚═════╝░ ╚═╝░░╚═╝ ╚═════╝░ ╚═╝ ░╚════╝░

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

/*≡==---==≡≡≡==---=≡≡*\
||    AFFINE EXPR    ||
\*≡==----==≡==----==≡*/

enum AffineBinaryOp(val symbol: String):
  override def toString = symbol

  case Add extends AffineBinaryOp("+")
  case Minus extends AffineBinaryOp("-")
  case Multiply extends AffineBinaryOp("*")
  case CeilDiv extends AffineBinaryOp("ceildiv")
  case FloorDiv extends AffineBinaryOp("floordiv")
  case Mod extends AffineBinaryOp("mod")

abstract class AffineExpr() {}

case class AffineBinaryOpExpr(
    val op: AffineBinaryOp,
    val lhs: AffineExpr,
    val rhs: AffineExpr
) extends AffineExpr {

  override def toString = s"${lhs} ${op} ${rhs}"
}

case class AffineDimExpr(val position: String) extends AffineExpr {

  override def toString = position
}

case class AffineSymExpr(val position: String) extends AffineExpr {

  override def toString = position
}

case class AffineConstantExpr(val value: Long) extends AffineExpr {

  override def toString = s"${value}"
}

/*≡==---==≡≡≡≡==---=≡≡*\
||     AFFINE MAP     ||
\*≡==----==≡≡==----==≡*/

case class AffineMap(
    val dimensions: Seq[String],
    val symbols: Seq[String],
    val affine_exprs: Seq[AffineExpr]
) {
  override def toString =
    s"(${dimensions.mkString(", ")})${symbols.mkString("[", ", ", "]")}" +
      s" -> (${affine_exprs.mkString(", ")})"
}

/*≡==---==≡≡≡≡==---=≡≡*\
||     AFFINE SET     ||
\*≡==----==≡≡==----==≡*/

enum AffineConstraintKind(val symbol: String):
  override def toString = symbol

  case GreaterEqual extends AffineConstraintKind(">=")
  case LessEqual extends AffineConstraintKind("<=")
  case Equal extends AffineConstraintKind("==")

case class AffineConstraintExpr(
    val kind: AffineConstraintKind,
    val lhs: AffineExpr,
    val rhs: AffineExpr
) {
  override def toString = s"${lhs} ${kind} ${rhs}"
}

case class AffineSet(
    val dimensions: Seq[String],
    val symbols: Seq[String],
    val affine_constraints: Seq[AffineConstraintExpr]
) {
  override def toString =
    s"(${dimensions.mkString(", ")})${symbols.mkString("[", ", ", "]")}" +
      s": (${affine_constraints.mkString(", ")})"
}

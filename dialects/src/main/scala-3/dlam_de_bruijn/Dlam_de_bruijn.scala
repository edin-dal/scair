package scair.dialects.dlam_de_bruijn

import scair.ir.*
import scair.dialects.builtin.*
import scair.clair.macros.*
import scair.clair.codegen.*
import scair.Printer
import scair.AttrParser
import fastparse.ParsingRun
import fastparse.*

/** \========================= Types (with de Bruijn) \=========================
  *
  * Model dlam types as Attributes to match the general scair.ir shape. If
  * codebase differentiates Type/Attribute at the kind level, change DlamType to
  * extend the correct base (e.g., Type) accordingly.
  */

// A sealed "kind" for all dlam types
sealed trait DlamType extends TypeAttribute

// !dlam.type  — the universe of dlam types
final case class DlamTypeType()
    extends DlamType
    with DerivedAttribute["dlam.type", DlamTypeType]
    derives DerivedAttributeCompanion

// !dlam.bvar<k>  — De Bruijn index (k is data)
final case class DlamBVarType(k: IntegerAttr)
    extends DlamType
    with DerivedAttribute["dlam.bvar", DlamBVarType]
    derives DerivedAttributeCompanion
//override def name: String = "dlam.bvar"
//override def parameters: Seq[Attribute | Seq[Attribute]] = Seq(k)

// !dlam.fun<in -> out> — function type
final case class DlamFunType(in: TypeAttribute, out: TypeAttribute)
    extends ParametrizedAttribute(),
      DlamType:
  override def name: String = "dlam.fun"
  override def parameters: Seq[Attribute | Seq[Attribute]] = Seq(in, out)

object DlamFunType extends AttributeCompanion:
  override def name = "dlam.fun"

  // override def parse[$: ParsingRun](p: AttrParser): ParsingRun[Attribute] = ???
  override def parse[$: P](p: AttrParser): P[DlamFunType] =
    import scair.AttrParser.whitespace
    P(
      "<" ~ p.Type ~ "," ~ p.Type ~ ">"
    ).map { (in, out) =>
      DlamFunType(
        in.asInstanceOf[TypeAttribute],
        out.asInstanceOf[TypeAttribute]
      )
    }

// !dlam.forall<body> — polymorphic type (body may reference bvar(0))
final case class DlamForAllType(body: TypeAttribute)
    extends DlamType
    with DerivedAttribute["dlam.forall", DlamForAllType]
    derives DerivedAttributeCompanion
//override def name: String = "dlam.forall"
//override def parameters: Seq[Attribute | Seq[Attribute]] = Seq(body)

// --------------------- Nat indexing ---------------------

// A tiny AST of natural-number expressions we can embed in types
sealed trait NatExpr extends TypeAttribute

final case class DlamNatLit(n: IntData)
    extends ParametrizedAttribute(),
      NatExpr:
  override def name = "dlam.nat_lit"
  override def parameters = Seq(n)

  override def custom_verify() =
    if n.value >= 0 then Right(()) else Left("nat_lit must be ≥ 0")

final case class DlamNatAdd(lhs: NatExpr, rhs: NatExpr)
    extends ParametrizedAttribute(),
      NatExpr:
  override def name = "dlam.nat.add"
  override def parameters = Seq(lhs, rhs)

final case class DlamNatMul(lhs: NatExpr, rhs: NatExpr)
    extends ParametrizedAttribute(),
      NatExpr:
  override def name = "dlam.nat.mul"
  override def parameters = Seq(lhs, rhs)

// !dlam.vec<len, elem> — length-indexed vector type
final case class DlamVecType(len: NatExpr, elem: TypeAttribute)
    extends ParametrizedAttribute(),
      DlamType:
  override def name: String = "dlam.vec"
  override def parameters: Seq[Attribute | Seq[Attribute]] = Seq(len, elem)

object DlamTy:
  inline def `type`: DlamType = DlamTypeType()
  inline def bvar(k: IntData): DlamType = DlamBVarType(IntegerAttr(k, I64))

  inline def fun(in: TypeAttribute, out: TypeAttribute): DlamType =
    DlamFunType(in, out)

  inline def forall(body: TypeAttribute): DlamType = DlamForAllType(body)

  inline def vec(len: NatExpr, elem: TypeAttribute): DlamType =
    DlamVecType(len, elem)

  // Nat Helpers:
  inline def n(n: Int): NatExpr = DlamNatLit(IntData(n))
  inline def nadd(a: NatExpr, b: NatExpr): NatExpr = DlamNatAdd(a, b)
  inline def nmul(a: NatExpr, b: NatExpr): NatExpr = DlamNatMul(a, b)

/** \========================= de Bruijn utilities \=========================
  *   - shift(d, c, t) — increase indices >= c by d (used when entering/leaving
  *     binders)
  *   - subst(c, s, t) — substitute BVar(c) in t with s (capture-avoiding)
  */
object DBI:
  import DlamTy.*

  // shift(d, c, t): increase all indices >= c by d
  def shift(d: Int, c: Int, t: TypeAttribute): TypeAttribute = t match
    case DlamBVarType(IntegerAttr(k, t)) if k.data >= c =>
      bvar(IntData(k.data + d))
    case b @ DlamBVarType(_)    => b
    case DlamFunType(i, o)      => fun(shift(d, c, i), shift(d, c, o))
    case DlamForAllType(body)   => forall(shift(d, c + 1, body))
    case DlamVecType(len, elem) => DlamVecType(len, shift(d, c, elem))
    case other                  => other

  // subst(c, s, t): substitute bvar(c) := s
  def subst(c: Int, s: TypeAttribute, t: TypeAttribute): TypeAttribute = t match
    case DlamBVarType(IntegerAttr(k, t)) if k.data == c => s
    case DlamBVarType(IntegerAttr(k, t)) if k.data > c  =>
      bvar(IntData(k.data - 1))
    case b @ DlamBVarType(_)  => b
    case DlamFunType(i, o)    => fun(subst(c, s, i), subst(c, s, o))
    case DlamForAllType(body) =>
      forall(subst(c + 1, shift(1, 0, s), body)) // or maybe shift by c?
    case DlamVecType(len, elem) =>
      DlamVecType(len, subst(c, s, elem))
    case other => other

  // instantiate ∀.body with arg
  def instantiate(fa: TypeAttribute, arg: TypeAttribute): TypeAttribute =
    fa match
      case DlamForAllType(body) => subst(0, arg, body)
      case other                => other

/** \========================= Operations \=========================
  *
  * Model the same surface as in examples:
  *   - tlambda / treturn / tapply (type-level abstraction/application)
  *   - vlambda / vreturn (value-level abstraction/return)
  * Regions carry the binders; the de Bruijn indices refer to region nesting.
  */

/** dlam.vlambda %v = dlam.vlambda (%x : A) : !dlam.fun<A -> B> { ... }
  * Invariants:
  *   - exactly one block with exactly one argument of type == funTyp.in
  *   - result type == funTyp
  */
final case class VLambda(
    funAttr: DlamType, // expect DlamFunType
    body: Region,
    res: Result[TypeAttribute]
) extends DerivedOperation["dlam.vlambda", VLambda]
    derives DerivedOperationCompanion:

  override def verify() =
    (funAttr, res.typ) match
      case (f @ DlamFunType(in, _), r) if r == f =>
        body.blocks match
          case Block(args, _) :: Nil
              if args.length == 1 && args.head.typ == in =>
            Right(this)
          case _ =>
            Left("vlambda: one block with one arg of input type required")
      case _ => Left("vlambda: result type must equal function type")

/** dlam.vreturn dlam.vreturn %x : T
  */
final case class VReturn(
    value: Value[TypeAttribute],
    expected: TypeAttribute
) extends DerivedOperation["dlam.vreturn", VReturn]
    with IsTerminator derives DerivedOperationCompanion:

  override def verify() =
    if value.typ == expected then Right(this)
    else Left("vreturn: type mismatch")

/** dlam.tlambda (∀-introduction) %F = dlam.tlambda (%T : !dlam.type) :
  * !dlam.forall< Body > { ... } Invariants:
  *   - exactly one block
  *   - that block has zero arguments
  *   - result type is ForAllType(_)
  */
final case class TLambda(
    tBody: Region,
    res: Result[TypeAttribute] // expect DlamForAllType
) extends DerivedOperation["dlam.tlambda", TLambda]
    derives DerivedOperationCompanion:

  override def verify() =
    val okBinder = tBody.blocks match
      case Block(args, _) :: Nil =>
        args.isEmpty
      case _ => false
    val okRes = res.typ.isInstanceOf[DlamForAllType]
    if okBinder && okRes then Right(this)
    else Left("tlambda: must have one block with zero args and a forall result")

/** dlam.treturn dlam.treturn %v : T
  */
final case class TReturn(
    value: Value[TypeAttribute],
    expected: TypeAttribute
) extends DerivedOperation["dlam.treturn", TReturn]
    with IsTerminator derives DerivedOperationCompanion:

  override def verify() =
    if value.typ == expected then Right(this)
    else Left("treturn: type mismatch")

/** dlam.tapply (∀-elimination) %h = dlam.tapply %G <!Ty> : InstantiatedType
  * Invariants:
  *   - polymorphicFun.typ == ForAllType(body)
  *   - res.typ == instantiate(ForAllType(body), argType) (de Bruijn-aware
  *     substitution)
  */
final case class TApply(
    polymorphicFun: Value[TypeAttribute], // expect DlamForAllType
    argType: TypeAttribute,
    res: Result[TypeAttribute]
) extends DerivedOperation["dlam.tapply", TApply]
    derives DerivedOperationCompanion:

  override def verify() =
    polymorphicFun.typ match
      case fa @ DlamForAllType(_) =>
        val inst = DBI.instantiate(fa, argType)
        if res.typ == inst then Right(this)
        else Left(s"tapply: result ${res.typ} != instantiated ${inst}")
      case other => Left(s"tapply: not a forall, got $other")

// Apply a value-level function to an argument: (%f %x) : out
final case class VApply(
    fun: Value[TypeAttribute],
    arg: Value[TypeAttribute],
    res: Result[TypeAttribute]
) extends DerivedOperation["dlam.vapply", VApply]
    derives DerivedOperationCompanion:

  override def verify() =
    fun.typ match
      case DlamFunType(in, out) =>
        if arg.typ == in && res.typ == out then Right(this)
        else
          Left(
            s"vapply: expected arg $in and result $out, got ${arg.typ} and ${res.typ}"
          )
      case other => Left(s"vapply: fun not a function type: $other")

/** \========================= Dialect Registration \=========================
  */
val DlamDialect = summonDialect[
  // Custom attributes
  (DlamTypeType, DlamBVarType, DlamForAllType),
  // Operations
  (VLambda, VReturn, TLambda, TReturn, TApply, VApply)
](Seq(DlamFunType))

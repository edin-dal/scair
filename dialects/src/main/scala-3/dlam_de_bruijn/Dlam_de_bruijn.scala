package scair.dialects.dlam_de_bruijn

import scair.ir.*
import scair.utils.*
import scair.dialects.builtin.*
import scair.clair.macros.*
import scair.clair.codegen.*
import scair.Printer
import scair.parse.*
import fastparse.ParsingRun
import fastparse.*

// ========================= Types (with de Bruijn) \=========================

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

// !dlam.fun<in -> out> — function type
final case class DlamFunType(in: TypeAttribute, out: TypeAttribute)
    extends ParametrizedAttribute(),
      DlamType:
  override def name: String = "dlam.fun"
  override def parameters: Seq[Attribute | Seq[Attribute]] = Seq(in, out)

given AttributeCompanion[DlamFunType]:
  override def name = "dlam.fun"

  override def parse[$: P](using Parser): P[DlamFunType] =
    P(
      "<" ~ typeP ~ "," ~ typeP ~ ">"
    ).map { (in, out) =>
      DlamFunType(
        in.asInstanceOf[TypeAttribute],
        out.asInstanceOf[TypeAttribute],
      )
    }

// !dlam.forall<body> — polymorphic type (body may reference bvar(0))
final case class DlamForAllType(body: TypeAttribute)
    extends DlamType
    with DerivedAttribute["dlam.forall", DlamForAllType]
    derives DerivedAttributeCompanion

object DlamTy:
  inline def `type`: DlamType = DlamTypeType()
  inline def bvar(k: IntData): DlamType = DlamBVarType(IntegerAttr(k, I64))

  inline def fun(in: TypeAttribute, out: TypeAttribute): DlamType =
    DlamFunType(in, out)

  inline def forall(body: TypeAttribute): DlamType = DlamForAllType(body)

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
    case b @ DlamBVarType(_)  => b
    case DlamFunType(i, o)    => fun(shift(d, c, i), shift(d, c, o))
    case DlamForAllType(body) => forall(shift(d, c + 1, body))
    case other                => other

  // subst(c, s, t): substitute bvar(c) := s
  def subst(c: Int, s: TypeAttribute, t: TypeAttribute): TypeAttribute = t match
    case DlamBVarType(IntegerAttr(k, t)) if k.data == c => s
    case DlamBVarType(IntegerAttr(k, t)) if k.data > c  =>
      bvar(IntData(k.data - 1))
    case b @ DlamBVarType(_)  => b
    case DlamFunType(i, o)    => fun(subst(c, s, i), subst(c, s, o))
    case DlamForAllType(body) =>
      forall(subst(c + 1, shift(1, 0, s), body)) // or maybe shift by c?
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
    res: Result[TypeAttribute],
) extends DerivedOperation["dlam.vlambda", VLambda]
    derives DerivedOperationCompanion:

  override def verify(): OK[Operation] =
    (funAttr, res.typ) match
      case (f @ DlamFunType(in, _), r) if r == f =>
        body.blocks match
          case Block(args, _) :: Nil
              if args.length == 1 && args.head.typ == in =>
            OK(this)
          case _ =>
            Err("vlambda: one block with one arg of input type required")
      case _ => Err("vlambda: result type must equal function type")

/** dlam.vreturn dlam.vreturn %x : T
  */
final case class VReturn(
    value: Value[TypeAttribute],
    expected: TypeAttribute,
) extends DerivedOperation["dlam.vreturn", VReturn]
    with IsTerminator derives DerivedOperationCompanion:

  override def verify(): OK[Operation] =
    if value.typ == expected then OK(this)
    else Err("vreturn: type mismatch")

/** dlam.tlambda (∀-introduction) %F = dlam.tlambda (%T : !dlam.type) :
  * !dlam.forall< Body > { ... } Invariants:
  *   - exactly one block
  *   - that block has zero arguments
  *   - result type is ForAllType(_)
  */
final case class TLambda(
    tBody: Region,
    res: Result[TypeAttribute], // expect DlamForAllType
) extends DerivedOperation["dlam.tlambda", TLambda]
    derives DerivedOperationCompanion:

  override def verify(): OK[Operation] =
    val okBinder = tBody.blocks match
      case Block(args, _) :: Nil =>
        args.isEmpty
      case _ => false
    val okRes = res.typ.isInstanceOf[DlamForAllType]
    if okBinder && okRes then OK(this)
    else Err("tlambda: must have one block with zero args and a forall result")

/** dlam.treturn dlam.treturn %v : T
  */
final case class TReturn(
    value: Value[TypeAttribute],
    expected: TypeAttribute,
) extends DerivedOperation["dlam.treturn", TReturn]
    with IsTerminator derives DerivedOperationCompanion:

  override def verify(): OK[Operation] =
    if value.typ == expected then OK(this)
    else Err("treturn: type mismatch")

/** dlam.tapply (∀-elimination) %h = dlam.tapply %G <!Ty> : InstantiatedType
  * Invariants:
  *   - polymorphicFun.typ == ForAllType(body)
  *   - res.typ == instantiate(ForAllType(body), argType) (de Bruijn-aware
  *     substitution)
  */
final case class TApply(
    polymorphicFun: Value[TypeAttribute], // expect DlamForAllType
    argType: TypeAttribute,
    res: Result[TypeAttribute],
) extends DerivedOperation["dlam.tapply", TApply]
    derives DerivedOperationCompanion:

  override def verify(): OK[Operation] =
    polymorphicFun.typ match
      case fa @ DlamForAllType(_) =>
        val inst = DBI.instantiate(fa, argType)
        if res.typ == inst then OK(this)
        else Err(s"tapply: result ${res.typ} != instantiated $inst")
      case other => Err(s"tapply: not a forall, got $other")

// Apply a value-level function to an argument: (%f %x) : out
final case class VApply(
    fun: Value[TypeAttribute],
    arg: Value[TypeAttribute],
    res: Result[TypeAttribute],
) extends DerivedOperation["dlam.vapply", VApply]
    derives DerivedOperationCompanion:

  override def verify(): OK[Operation] =
    fun.typ match
      case DlamFunType(in, out) =>
        if arg.typ == in && res.typ == out then OK(this)
        else
          Err(
            s"vapply: expected arg $in and result $out, got ${arg.typ} and ${res
                .typ}"
          )
      case other => Err(s"vapply: fun not a function type: $other")

// ========================= Dialect Registration \=========================
val DlamDeBruijnDialect = summonDialect[
  // Custom attributes
  (DlamTypeType, DlamBVarType, DlamForAllType, DlamFunType),
  // Operations
  (VLambda, VReturn, TLambda, TReturn, TApply, VApply),
]

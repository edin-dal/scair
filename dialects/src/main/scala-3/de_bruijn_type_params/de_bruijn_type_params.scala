package scair.dialects.de_bruijn_type_params

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

// A sealed "kind" for all tlam types
sealed trait tlamType extends TypeAttribute

// !tlam.type  — the universe of tlam types
final case class tlamTypeType()
    extends tlamType
    with DerivedAttribute["tlam.type", tlamTypeType]
    derives DerivedAttributeCompanion

// !tlam.bvar<k>  — De Bruijn index (k is data)
final case class tlamBVarType(k: IntegerAttr)
    extends tlamType
    with DerivedAttribute["tlam.bvar", tlamBVarType]
    derives DerivedAttributeCompanion

// !tlam.fun<in -> out> — function type
final case class tlamFunType(in: TypeAttribute, out: TypeAttribute)
    extends ParametrizedAttribute(),
      tlamType:
  override def name: String = "tlam.fun"
  override def parameters: Seq[Attribute | Seq[Attribute]] = Seq(in, out)

given AttributeCompanion[tlamFunType]:
  override def name = "tlam.fun"

  override def parse[$: P](using Parser): P[tlamFunType] =
    P(
      "<" ~ typeP ~ "," ~ typeP ~ ">"
    ).map { (in, out) =>
      tlamFunType(
        in.asInstanceOf[TypeAttribute],
        out.asInstanceOf[TypeAttribute],
      )
    }

// !tlam.forall<body> — polymorphic type (body may reference bvar(0))
final case class tlamForAllType(body: TypeAttribute)
    extends tlamType
    with DerivedAttribute["tlam.forall", tlamForAllType]
    derives DerivedAttributeCompanion

object tlamTy:
  inline def `type`: tlamType = tlamTypeType()
  inline def bvar(k: IntData): tlamType = tlamBVarType(IntegerAttr(k, I64))

  inline def fun(in: TypeAttribute, out: TypeAttribute): tlamType =
    tlamFunType(in, out)

  inline def forall(body: TypeAttribute): tlamType = tlamForAllType(body)

/** \========================= de Bruijn utilities \=========================
  *   - shift(d, c, t) — increase indices >= c by d (used when entering/leaving
  *     binders)
  *   - subst(c, s, t) — substitute BVar(c) in t with s (capture-avoiding)
  */
object DBI:
  import tlamTy.*

  // shift(d, c, t): increase all indices >= c by d
  def shift(d: Int, c: Int, t: TypeAttribute): TypeAttribute = t match
    case tlamBVarType(IntegerAttr(k, t)) if k.data >= c =>
      bvar(IntData(k.data + d))
    case b @ tlamBVarType(_)  => b
    case tlamFunType(i, o)    => fun(shift(d, c, i), shift(d, c, o))
    case tlamForAllType(body) => forall(shift(d, c + 1, body))
    case other                => other

  // subst(c, s, t): substitute bvar(c) := s
  def subst(c: Int, s: TypeAttribute, t: TypeAttribute): TypeAttribute = t match
    case tlamBVarType(IntegerAttr(k, t)) if k.data == c => s
    case tlamBVarType(IntegerAttr(k, t)) if k.data > c  =>
      bvar(IntData(k.data - 1))
    case b @ tlamBVarType(_)  => b
    case tlamFunType(i, o)    => fun(subst(c, s, i), subst(c, s, o))
    case tlamForAllType(body) =>
      forall(subst(c + 1, shift(1, 0, s), body)) // or maybe shift by c?
    case other => other

  // instantiate ∀.body with arg
  def instantiate(fa: TypeAttribute, arg: TypeAttribute): TypeAttribute =
    fa match
      case tlamForAllType(body) => subst(0, arg, body)
      case other                => other

/** \========================= Operations \=========================
  *
  * Model the same surface as in examples:
  *   - tlambda / treturn / tapply (type-level abstraction/application)
  *   - vlambda / vreturn (value-level abstraction/return)
  * Regions carry the binders; the de Bruijn indices refer to region nesting.
  */

/** tlam.vlambda %v = tlam.vlambda (%x : A) : !tlam.fun<A -> B> { ... }
  * Invariants:
  *   - exactly one block with exactly one argument of type == funTyp.in
  *   - result type == funTyp
  */
final case class VLambda(
    funAttr: tlamType, // expect tlamFunType
    body: Region,
    res: Result[TypeAttribute],
) extends DerivedOperation["tlam.vlambda", VLambda]
    derives DerivedOperationCompanion:

  override def verify(): OK[Operation] =
    (funAttr, res.typ) match
      case (f @ tlamFunType(in, _), r) if r == f =>
        body.blocks match
          case Block(args, _) :: Nil
              if args.length == 1 && args.head.typ == in =>
            OK(this)
          case _ =>
            Err("vlambda: one block with one arg of input type required")
      case _ => Err("vlambda: result type must equal function type")

/** tlam.vreturn tlam.vreturn %x : T
  */
final case class VReturn(
    value: Value[TypeAttribute],
    expected: TypeAttribute,
) extends DerivedOperation["tlam.vreturn", VReturn]
    with IsTerminator derives DerivedOperationCompanion:

  override def verify(): OK[Operation] =
    if value.typ == expected then OK(this)
    else Err("vreturn: type mismatch")

/** tlam.tlambda (∀-introduction) %F = tlam.tlambda (%T : !tlam.type) :
  * !tlam.forall< Body > { ... } Invariants:
  *   - exactly one block
  *   - that block has zero arguments
  *   - result type is ForAllType(_)
  */
final case class TLambda(
    tBody: Region,
    res: Result[TypeAttribute], // expect tlamForAllType
) extends DerivedOperation["tlam.tlambda", TLambda]
    derives DerivedOperationCompanion:

  override def verify(): OK[Operation] =
    val okBinder = tBody.blocks match
      case Block(args, _) :: Nil =>
        args.isEmpty
      case _ => false
    val okRes = res.typ.isInstanceOf[tlamForAllType]
    if okBinder && okRes then OK(this)
    else Err("tlambda: must have one block with zero args and a forall result")

/** tlam.treturn tlam.treturn %v : T
  */
final case class TReturn(
    value: Value[TypeAttribute],
    expected: TypeAttribute,
) extends DerivedOperation["tlam.treturn", TReturn]
    with IsTerminator derives DerivedOperationCompanion:

  override def verify(): OK[Operation] =
    if value.typ == expected then OK(this)
    else Err("treturn: type mismatch")

/** tlam.tapply (∀-elimination) %h = tlam.tapply %G <!Ty> : InstantiatedType
  * Invariants:
  *   - polymorphicFun.typ == ForAllType(body)
  *   - res.typ == instantiate(ForAllType(body), argType) (de Bruijn-aware
  *     substitution)
  */
final case class TApply(
    polymorphicFun: Value[TypeAttribute], // expect tlamForAllType
    argType: TypeAttribute,
    res: Result[TypeAttribute],
) extends DerivedOperation["tlam.tapply", TApply]
    derives DerivedOperationCompanion:

  override def verify(): OK[Operation] =
    polymorphicFun.typ match
      case fa @ tlamForAllType(_) =>
        val inst = DBI.instantiate(fa, argType)
        if res.typ == inst then OK(this)
        else Err(s"tapply: result ${res.typ} != instantiated $inst")
      case other => Err(s"tapply: not a forall, got $other")

// Apply a value-level function to an argument: (%f %x) : out
final case class VApply(
    fun: Value[TypeAttribute],
    arg: Value[TypeAttribute],
    res: Result[TypeAttribute],
) extends DerivedOperation["tlam.vapply", VApply]
    derives DerivedOperationCompanion:

  override def verify(): OK[Operation] =
    fun.typ match
      case tlamFunType(in, out) =>
        if arg.typ == in && res.typ == out then OK(this)
        else
          Err(
            s"vapply: expected arg $in and result $out, got ${arg.typ} and ${res
                .typ}"
          )
      case other => Err(s"vapply: fun not a function type: $other")

// ========================= Dialect Registration \=========================
val DeBruijnTypeParamsDialect = summonDialect[
  // Custom attributes
  (tlamTypeType, tlamBVarType, tlamForAllType, tlamFunType),
  // Operations
  (VLambda, VReturn, TLambda, TReturn, TApply, VApply),
]

package scair.dialects.tlam_de_bruijn

import scair.ir.*
import scair.utils.*
import scair.dialects.builtin.*
import scair.clair.macros.*
import scair.clair.codegen.*
import scair.Printer
import scair.parse.*
import fastparse.ParsingRun
import fastparse.*

// TODO:
// - Use tlamType in forAllType?
// - change IntAttr to IntData?

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
    derives DerivedAttributeCompanion:

  override def customVerify(): OK[Unit] =
    if k.value < 0 then Err(s"tlam.bvar index must be >= 0, got ${k.value}")
    else OK(())

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
  inline def bvar(k: IntData): tlamBVarType = tlamBVarType(IntegerAttr(k, I64))

  inline def fun(in: TypeAttribute, out: TypeAttribute): tlamFunType =
    tlamFunType(in, out)

  inline def forall(body: TypeAttribute): tlamForAllType = tlamForAllType(body)

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
      forall(subst(c + 1, shift(1, 0, s), body))
    case other => other

  // instantiate forAll.body with arg
  def instantiate(fa: TypeAttribute, arg: TypeAttribute): TypeAttribute =
    fa match
      case tlamForAllType(body) => subst(0, arg, body)
      case other                => other

/** ========================= de Bruijn utilities ==========================
  *   - shift(d, c, t) — increase indices >= c by d (used when entering/leaving
  *     binders)
  *   - subst(c, s, t) — substitute BVar(c) in t with s (capture-avoiding)
  *
  * NOTE (tightened kinds):
  *   - `shift` traverses only well-formed tlam type terms (`tlamType`) and
  *     returns `tlamType`.
  *   - `subst` may *inject* an arbitrary `TypeAttribute` (e.g. `si32`) for
  *     `bvar(c)`, therefore it returns `TypeAttribute`.
  *   - Helpers `shiftAny`/`substAny` lift traversal over general `TypeAttribute`
  *     fields (e.g. fun in/out) by only recursing when the value is a `tlamType`.
  */
/*
object DBI:
  import tlamTy.*

  /** Shift a general type attribute: only tlam types contain bvars. */
  private def shiftAny(d: Int, c: Int, t: TypeAttribute): TypeAttribute =
    t match
      case tt: tlamType => shift(d, c, tt)
      case other        => other

  /** Substitute inside a general type attribute: only tlam types contain bvars. */
  private def substAny(c: Int, s: TypeAttribute, t: TypeAttribute): TypeAttribute =
    t match
      case tt: tlamType => subst(c, s, tt)
      case other        => other

  /** shift(d, c, t): increase all indices >= c by d */
  def shift(d: Int, c: Int, t: tlamType): tlamType = t match
    case tlamBVarType(IntegerAttr(k, _)) if k.data >= c =>
      bvar(IntData(k.data + d))
    case b @ tlamBVarType(_) =>
      b
    case tlamFunType(i, o) =>
      fun(shiftAny(d, c, i), shiftAny(d, c, o))
    case tlamForAllType(body) =>
      // Entering a binder: cutoff increases.
      forall(shift(d, c + 1, body))
    case other =>
      other

  /** subst(c, s, t): substitute bvar(c) := s
 *
    * Returns `TypeAttribute` because `s` may be a non-tlam (e.g. builtin) type.
 */
  def subst(c: Int, s: TypeAttribute, t: tlamType): TypeAttribute = t match
    case tlamBVarType(IntegerAttr(k, _)) if k.data == c =>
      s
    case tlamBVarType(IntegerAttr(k, _)) if k.data > c =>
      // One binder removed by substitution: decrement indices above the hole.
      bvar(IntData(k.data - 1))
    case b @ tlamBVarType(_) =>
      b
    case tlamFunType(i, o) =>
      fun(substAny(c, s, i), substAny(c, s, o))
    case tlamForAllType(body) =>
      // Under a binder:
      //  - increase cutoff
      //  - shift `s` to avoid capture (only affects tlam types)
      val s1 = shiftAny(1, 0, s)
      // This returns a TypeAttribute, but for the forall body it will remain a tlamType
      // for the intended use (instantiate uses c=0).
      forall(subst(c + 1, s1, body).asInstanceOf[tlamType])
    case other =>
      other

  /** Instantiate forall body with `arg` (i.e. substitute bvar(0) := arg). */
  def instantiate(fa: tlamForAllType, arg: TypeAttribute): TypeAttribute =
    subst(0, arg, fa.body)
 */

/** ========================= Operations =========================
  *
  * This dialect models a System F–style lambda calculus using
  * de Bruijn indices for type variables.
  *
  * Operations:
  *   - TLambda / TReturn / TApply : type-level abstraction, return, and application
  *   - VLambda / VReturn / VApply : value-level abstraction, return, and application
  *
  * Type binders are introduced implicitly by region nesting
  * (via TLambda). De Bruijn indices (`!tlam.bvar<k>`) refer to the
  * number of enclosing TLambda regions.
  *
  * Note:
  *   - Structural invariants are checked by each op’s `verify()`.
  *   - Well-scopedness of de Bruijn indices is enforced by a
  *     dedicated verifier pass, not here.
  */

/** tlam.vlambda — value-level lambda abstraction.
  *
  * Introduces a value-level binder via a single block argument.
  *
  * The operation represents a function of type `!tlam.fun<in, out>`, whose body
  * computes a value of type `out`.
  *
  * Local invariants:
  *   - the body consists of exactly one block
  *   - the block has exactly one argument of type `in`
  *   - the block is terminated by `tlam.vreturn`
  *   - the returned value has type `out`
  */

final case class VLambda(
    body: Region,
    res: Result[tlamFunType],
) extends DerivedOperation["tlam.vlambda", VLambda]
    derives DerivedOperationCompanion:

  override def verify(): OK[Operation] =
    val funTy = res.typ
    body.blocks match
      case Block(args, ops) :: Nil
          if args.length == 1 && args.head.typ == funTy.in =>
        ops.lastOption match
          case Some(VReturn(ret)) =>
            if ret.typ == funTy.out then OK(this)
            else
              Err(s"vlambda: return type mismatch, expected ${funTy
                  .out}, got ${ret.typ}")
          case Some(other) =>
            Err(s"vlambda: last op must be tlam.vreturn, got '${other.name}'")
          case None =>
            Err("vlambda: body block must not be empty (needs a terminator)")
      case _ =>
        Err("vlambda: one block with one arg of input type required")

final case class VReturn(
    value: Value[TypeAttribute]
) extends DerivedOperation["tlam.vreturn", VReturn]
    with IsTerminator derives DerivedOperationCompanion

/** tlam.tlambda — type-level lambda abstraction (forall introduction).
  *
  * Introduces one type binder implicitly via region nesting (de Bruijn
  * encoding).
  *
  * Inside the region:
  *   - the bound type variable is referenced as `!tlam.bvar<0>`
  *   - outer binders are referenced as `!tlam.bvar<1>`, etc.
  *
  * The operation produces a value of type `!tlam.forall<body>`, where `body` is
  * the type returned from the region.
  *
  * Local invariants:
  *   - exactly one block
  *   - the block has zero arguments
  *   - the block is terminated by `tlam.treturn`
  *   - the returned type equals the `forall` body
  *
  * Note:
  *   - Correct scoping of de Bruijn indices is checked by a separate verifier
  *     pass.
  */

final case class TLambda(
    body: Region,
    res: Result[tlamForAllType],
) extends DerivedOperation["tlam.tlambda", TLambda]
    derives DerivedOperationCompanion:

  override def verify(): OK[Operation] =
    body.blocks match
      case Block(args, ops) :: Nil if args.isEmpty =>
        ops.lastOption match
          case Some(TReturn(ret)) =>
            val expected = res.typ.body
            if ret.typ == expected then OK(this)
            else
              Err(
                s"tlambda: return type mismatch, expected $expected, got ${ret
                    .typ}"
              )
          case Some(other) =>
            Err(
              s"tlambda: last op must be tlam.treturn, got '${other.name}'"
            )
          case None =>
            Err("tlambda: body block must not be empty (needs a terminator)")
      case _ =>
        Err("tlambda: must have exactly one block with zero args")

final case class TReturn(
    value: Value[TypeAttribute]
) extends DerivedOperation["tlam.treturn", TReturn]
    with IsTerminator derives DerivedOperationCompanion

/** tlam.tapply — type application (forall elimination).
  *
  * Instantiates a polymorphic value of type `!tlam.forall<body>` with a
  * concrete type argument.
  *
  * The result type is obtained by de Bruijn-aware substitution of the argument
  * for `!tlam.bvar<0>` in `body`.
  *
  * Local invariants:
  *   - the operand has a `tlam.forall` type
  *   - the result type equals `instantiate(body, tyArg)`
  *
  * Note:
  *   - This operation assumes well-scoped de Bruijn indices.
  */

final case class TApply(
    fun: Value[tlamForAllType],
    tyArg: TypeAttribute,
    res: Result[TypeAttribute],
) extends DerivedOperation["tlam.tapply", TApply]
    derives DerivedOperationCompanion:

  override def verify(): OK[Operation] =
    val inst = DBI.instantiate(fun.typ, tyArg)
    if res.typ == inst then OK(this)
    else Err(s"tapply: result ${res.typ} != instantiated $inst")

/** tlam.vapply — value-level function application.
  *
  * Applies a value-level function of type `!tlam.fun<in, out>` to an argument
  * of type `in`, producing a result of type `out`.
  *
  * Local invariants:
  *   - `arg.typ == fun.typ.in`
  *   - `res.typ == fun.typ.out`
  */
final case class VApply(
    fun: Value[tlamFunType],
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

// ========================= Dialect Registration \=========================
val TlamDeBruijnDialect = summonDialect[
  // Custom attributes
  (tlamTypeType, tlamBVarType, tlamForAllType, tlamFunType),
  // Operations
  (VLambda, VReturn, TLambda, TReturn, TApply, VApply),
]

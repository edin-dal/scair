package scair.constraints

import scair.ir.*
import scair.utils.*

import scala.quoted.*

// ░█████╗░ ░█████╗░ ███╗░░██╗ ░██████╗ ████████╗ ██████╗░ ░█████╗░ ██╗ ███╗░░██╗ ████████╗ ░██████╗
// ██╔══██╗ ██╔══██╗ ████╗░██║ ██╔════╝ ╚══██╔══╝ ██╔══██╗ ██╔══██╗ ██║ ████╗░██║ ╚══██╔══╝ ██╔════╝
// ██║░░╚═╝ ██║░░██║ ██╔██╗██║ ╚█████╗░ ░░░██║░░░ ██████╔╝ ███████║ ██║ ██╔██╗██║ ░░░██║░░░ ╚█████╗░
// ██║░░██╗ ██║░░██║ ██║╚████║ ░╚═══██╗ ░░░██║░░░ ██╔══██╗ ██╔══██║ ██║ ██║╚████║ ░░░██║░░░ ░╚═══██╗
// ╚█████╔╝ ╚█████╔╝ ██║░╚███║ ██████╔╝ ░░░██║░░░ ██║░░██║ ██║░░██║ ██║ ██║░╚███║ ░░░██║░░░ ██████╔╝
// ░╚════╝░ ░╚════╝░ ╚═╝░░╚══╝ ╚═════╝░ ░░░╚═╝░░░ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ╚═╝ ╚═╝░░╚══╝ ░░░╚═╝░░░ ╚═════╝░

/** Constraints are *phantom* types: they describe a check, they are never
  * instantiated, and they leave no trace at run time.
  *
  * A constraint is attached to an operation's operand, result or property
  * through [[!>]], and compiled by its own [[ConstraintGen]] -- carried, by
  * convention, on its companion object -- into straight-line code inside the
  * operation's generated `constraintVerify`.
  *
  * The constraints below are the ones ScaIR ships, and they are not privileged:
  * each is a trait plus a companion generator, which is exactly what a
  * constraint defined in a downstream dialect is. The interpreter in
  * `ConstraintsMacros.scala` knows none of them by name; it only dispatches.
  *
  * Most constraints a dialect needs are combinations of these, and a
  * combination needs no generator at all -- a type alias is enough:
  *
  * ```scala
  * ///{
  *
  * import scair.constraints.*
  * import scair.dialects.builtin.*
  *
  * val f32 = Float32Type()
  * ///}
  * type F32 = EqAttr[f32.type]
  * type AnyFloat = Base[Float32Type] || Base[Float64Type]
  * type SignlessInt = Param[IntegerType, (AnyAttr, EqAttr[Signless.type])]
  * ```
  *
  * Deliberately not sealed: out-of-tree constraints extend it.
  */
trait Constraint

/** Attaches constraint `C` to attribute type `A`.
  *
  * Erases to `A`, so it is invisible everywhere except to the macros that read
  * the field's type.
  */
infix type !>[A <: Attribute, C <: Constraint] = A

/*≡==--==≡≡≡≡≡==--=≡≡*\
||       ATOMS       ||
\*≡==---==≡≡≡==---==≡*/

/** Satisfied by anything. Identity of [[AllOf]], zero of [[AnyOf]]. */
trait AnyAttr extends Constraint

object AnyAttr extends ConstraintGen:

  def verify(c: Type[?], attr: Expr[Attribute], path: String)(using
      Quotes,
      GenCtx,
      ErrCtx,
  ) = None // Nothing to check, so nothing to emit.

/** Satisfied by any attribute that is an `A`. */
trait Base[A <: Attribute] extends Constraint

object Base extends ConstraintGen:

  def verify(c: Type[?], attr: Expr[Attribute], path: String)(using
      Quotes,
      GenCtx,
      ErrCtx,
  ) =
    c match
      case '[Base[a]] =>
        val name = Expr(Type.show[a])
        Some('{
          if $attr.isInstanceOf[a] then OK()
          else ${ fail(path, name, attr) }
        })

/** Satisfied only by the attribute `A` denotes.
  *
  * `A` must be a singleton type of a stable value, i.e. `f32.type` for some
  * `val f32 = Float32Type()`.
  */
trait EqAttr[A <: Attribute] extends Constraint

object EqAttr extends ConstraintGen:

  def verify(c: Type[?], attr: Expr[Attribute], path: String)(using
      Quotes,
      GenCtx,
      ErrCtx,
  ) =
    c match
      case '[EqAttr[a]] =>
        val ref = attributeOf[a]
        Some('{
          if $attr == $ref then OK()
          else ${ fail(path, '{ $ref.toString }, attr) }
        })

  /** One attribute satisfies this, so it is always inferable. */
  override def infer(c: Type[?])(using Quotes, GenCtx) =
    c match
      case '[EqAttr[a]] => Some(attributeOf[a])

/** A named constraint variable, scoped to one operation.
  *
  * Its first occurrence binds; every later occurrence requires equality with
  * what was bound. Two operands sharing a `Var` must have equal types, and a
  * result sharing it with an operand is MLIR's `SameOperandsAndResultType`.
  *
  * Sharing a variable is free: the first occurrence emits no code at all, it
  * just records *where the value can be read back from*, so a later occurrence
  * compiles to a direct comparison against that path.
  */
trait Var[N <: String] extends Constraint

object Var extends ConstraintGen:

  private def nameOf(c: Type[?])(using Quotes): String =
    c match
      case '[Var[n]] => Type.valueOfConstant[n].get.asInstanceOf[String]

  def verify(c: Type[?], attr: Expr[Attribute], path: String)(using
      q: Quotes,
      ctx: GenCtx,
      err: ErrCtx,
  ) =
    val name = nameOf(c)
    ctx.lookup(name) match
      case None =>
        ctx.bind(name, attr) match
          // Bound at a position that can be named later: nothing to emit, we
          // just remember where to read the value back from.
          case Binding(_, None) => None
          // Bound in a slot, so the binding happens at run time.
          case Binding(_, Some(assign)) => Some('{ ${ assign(attr) }; OK() })

      case Some(Binding(read, None)) =>
        Some('{
          if $attr == $read then OK()
          else ${ fail(path, '{ $read.toString }, attr) }
        })

      case Some(Binding(read, Some(assign))) =>
        // Whichever occurrence executes first fills the slot, the rest check
        // against it. xDSL's `VarConstraint.verify`, over a local instead of a
        // dictionary.
        Some('{
          if $read == null then
            ${ assign(attr) }
            OK()
          else if $read != $attr then ${ fail(path, '{ $read.toString }, attr) }
          else OK()
        })

  override def infer(c: Type[?])(using Quotes, GenCtx) =
    summon[GenCtx].lookup(nameOf(c)).map(_.read)

  override def bind(c: Type[?], attr: Expr[Attribute])(using Quotes, GenCtx) =
    val ctx = summon[GenCtx]
    val name = nameOf(c)
    if ctx.lookup(name).isEmpty then ctx.bind(name, attr)

/** Satisfied by an `A` whose parameters satisfy `Cs` pairwise.
  *
  * `A` must be a case class and `Cs` must have one constraint per case field.
  */
trait Param[A <: Attribute, Cs <: Tuple] extends Constraint

object Param extends ConstraintGen:

  /** The case fields of `A` paired with their constraints, arity-checked. */
  private def fields[A: Type](constrs: List[Type[?]])(using
      Quotes
  ): List[(quotes.reflect.Symbol, Type[?])] =
    import quotes.reflect.*
    val fs = TypeRepr.of[A].typeSymbol.caseFields
    def name = Type.show[A]
    if fs.isEmpty then
      report.errorAndAbort(
        s"Param[$name, ...] requires $name to be a case class with parameters."
      )
    if fs.length != constrs.length then
      report
        .errorAndAbort(
          s"Param[$name, ...] expects ${fs.length} constraints " +
            s"(${fs.map(_.name).mkString(", ")}), got ${constrs.length}."
        )
    fs.zip(constrs)

  def verify(c: Type[?], attr: Expr[Attribute], path: String)(using
      Quotes,
      GenCtx,
      ErrCtx,
  ) =
    c match
      case '[type a <: Attribute; Param[`a`, cs]] =>
        val pairs = fields[a](tupleTypes[cs])
        val name = Expr(Type.show[a])
        Some('{
          $attr match
            case p: a =>
              ${
                // A parameter is reached through a pattern binding that does
                // not exist further down the method, so a variable bound in
                // here needs a slot.
                summon[GenCtx].unnameable {
                  val checks = pairs.flatMap { (field, fc) =>
                    verifyC(
                      fc,
                      selectMember[Attribute]('p, field.name),
                      s"$path parameter '${field.name}'",
                    )
                  }
                  chain(checks).getOrElse('{ OK() })
                }
              }
            case other => ${ fail(path, name, 'other) }
        })

  override def infer(c: Type[?])(using Quotes, GenCtx) =
    import quotes.reflect.*
    c match
      case '[type a <: Attribute; Param[`a`, cs]] =>
        val pairs = fields[a](tupleTypes[cs])
        val args = pairs.map((_, fc) => inferC(fc))
        if args.exists(_.isEmpty) then None
        else
          // Build `new A(p1, ..., pn)`. The inferred arguments are only known
          // to be Attributes, so each is cast to its parameter's declared type;
          // a mismatch is exactly what this constraint rejects at verification
          // time.
          val tpe = TypeRepr.of[a]
          val cast = pairs.zip(args.map(_.get)).map { case ((field, _), arg) =>
            tpe.memberType(field).asType match
              case '[t] => '{ $arg.asInstanceOf[t] }.asTerm
          }
          Some(
            Apply(
              Select(New(TypeTree.of[a]), tpe.typeSymbol.primaryConstructor),
              cast,
            ).asExprOf[Attribute]
          )

  override def bind(c: Type[?], attr: Expr[Attribute])(using Quotes, GenCtx) =
    c match
      case '[type a <: Attribute; Param[`a`, cs]] =>
        // Unlike `verify`, `attr` here is an ordinary expression rather than a
        // pattern binding, so its parameters can be named directly.
        val cast = '{ $attr.asInstanceOf[a] }
        fields[a](tupleTypes[cs]).foreach { (field, fc) =>
          bindC(fc, selectMember[Attribute](cast, field.name))
        }

/** A constraint built out of other constraints.
  *
  * Recursing into sub-constraints is the same three methods every time, so
  * composites only have to say what their parts are, and how to check them.
  * That keeps them in step as [[ConstraintGen]] grows, and gives downstream
  * authors of composite constraints the same base the built-ins use.
  */
trait CompositeGen extends ConstraintGen:

  /** The constraints this one is built from. */
  def parts(c: Type[?])(using Quotes): List[Type[?]]

  override def infer(c: Type[?])(using Quotes, GenCtx) =
    parts(c).iterator.flatMap(inferC).nextOption()

  override def bind(c: Type[?], attr: Expr[Attribute])(using Quotes, GenCtx) =
    parts(c).foreach(bindC(_, attr))

/** `C`, but reporting `M` instead of `C`'s own message on failure. */
trait Msg[M <: String, C <: Constraint] extends Constraint

object Msg extends CompositeGen:

  def parts(c: Type[?])(using Quotes) =
    c match
      case '[Msg[m, inner]] => List(Type.of[inner])

  def verify(c: Type[?], attr: Expr[Attribute], path: String)(using
      Quotes,
      GenCtx,
      ErrCtx,
  ) =
    c match
      case '[Msg[m, inner]] =>
        given ErrCtx =
          ErrCtx(Some(Type.valueOfConstant[m].get.asInstanceOf[String]))
        verifyC(Type.of[inner], attr, path)

/*≡==--==≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||      COMBINATORS       ||
\*≡==---==≡≡≡≡≡≡≡≡==---==≡*/

/** Satisfied when every constraint in `Cs` is. Checks short-circuit. */
trait AllOf[Cs <: Tuple] extends Constraint

object AllOf extends CompositeGen:

  def parts(c: Type[?])(using Quotes) =
    c match
      case '[AllOf[cs]] => tupleTypes[cs]

  def verify(c: Type[?], attr: Expr[Attribute], path: String)(using
      Quotes,
      GenCtx,
      ErrCtx,
  ) = chain(parts(c).flatMap(verifyC(_, attr, path)))

/** Satisfied when some constraint in `Cs` is; alternatives are tried in order.
  *
  * Unlike xDSL's `AnyOf`, alternatives need not have disjoint or even known
  * base types: the generated code short-circuits on the first success and rolls
  * back any constraint variable a failed alternative bound.
  */
trait AnyOf[Cs <: Tuple] extends Constraint

object AnyOf extends CompositeGen:

  def parts(c: Type[?])(using Quotes) =
    c match
      case '[AnyOf[cs]] => tupleTypes[cs]

  def verify(c: Type[?], attr: Expr[Attribute], path: String)(using
      q: Quotes,
      ctx: GenCtx,
      err: ErrCtx,
  ) =
    val alts = parts(c)

    // An alternative may not run, so a variable bound inside one binds at run
    // time; and an alternative that fails must not leave that binding behind,
    // or the next would be checked against a value nothing agreed on. xDSL
    // sidesteps this by forbidding disjunctions that can bind at all; we track
    // exactly the slots these alternatives mint and put those back.
    val (checks, slots) =
      ctx.tracking(ctx.unnameable(alts.map(verifyC(_, attr, path))))

    val expected =
      alts.map(a => Type.show(using a)).mkString("one of ", " | ", "")

    /** `if <alt> then done else { restore this alternative's slots; <rest> }`
      */
    def alternative(
        check: Expr[OK[Unit]],
        rest: Expr[OK[Unit]],
        cells: List[Binding],
    ): Expr[OK[Unit]] =
      cells match
        case Nil     => '{ val r = $check; if r.isOK then r else $rest }
        case b :: bs =>
          val restore = b.assign.get
          '{
            val saved = ${ b.read }
            ${ alternative(check, '{ ${ restore('saved) }; $rest }, bs) }
          }

    Some(
      checks.foldRight(fail(path, Expr(expected), attr))((c, rest) =>
        c match
          case None        => '{ OK() } // an alternative that always holds
          case Some(check) => alternative(check, rest, slots.toList)
      )
    )

  // Not inferable: more than one attribute satisfies a disjunction. And a
  // variable bound in only some alternatives is not bound at all -- xDSL's rule
  // too, whose `AnyOf.variables()` intersects its alternatives -- so `bind`,
  // which seeds inference, must not descend into them either.
  override def infer(c: Type[?])(using Quotes, GenCtx) = None

  override def bind(c: Type[?], attr: Expr[Attribute])(using Quotes, GenCtx) =
    ()

infix type &&[A <: Constraint, B <: Constraint] = AllOf[(A, B)]

infix type ||[A <: Constraint, B <: Constraint] = AnyOf[(A, B)]

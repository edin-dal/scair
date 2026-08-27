package scair.constraints

import scair.ir.*
import scair.utils.*

import scala.collection.mutable
import scala.quoted.*

//
// ░█████╗░ ░█████╗░ ███╗░░██╗ ░██████╗ ████████╗ ██████╗░ ░█████╗░ ██╗ ███╗░░██╗ ████████╗ ░██████╗
// ██╔══██╗ ██╔══██╗ ████╗░██║ ██╔════╝ ╚══██╔══╝ ██╔══██╗ ██╔══██╗ ██║ ████╗░██║ ╚══██╔══╝ ██╔════╝
// ██║░░╚═╝ ██║░░██║ ██╔██╗██║ ╚█████╗░ ░░░██║░░░ ██████╔╝ ███████║ ██║ ██╔██╗██║ ░░░██║░░░ ╚█████╗░
// ██║░░██╗ ██║░░██║ ██║╚████║ ░╚═══██╗ ░░░██║░░░ ██╔══██╗ ██╔══██║ ██║ ██║╚████║ ░░░██║░░░ ░╚═══██╗
// ╚█████╔╝ ╚█████╔╝ ██║░╚███║ ██████╔╝ ░░░██║░░░ ██║░░██║ ██║░░██║ ██║ ██║░╚███║ ░░░██║░░░ ██████╔╝
// ░╚════╝░ ░╚════╝░ ╚═╝░░╚══╝ ╚═════╝░ ░░░╚═╝░░░ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ╚═╝ ╚═╝░░╚══╝ ░░░╚═╝░░░ ╚═════╝░
//
// ███╗░░░███╗ ░█████╗░ ░█████╗░ ██████╗░ ░█████╗░ ░██████╗
// ████╗░████║ ██╔══██╗ ██╔══██╗ ██╔══██╗ ██╔══██╗ ██╔════╝
// ██╔████╔██║ ███████║ ██║░░╚═╝ ██████╔╝ ██║░░██║ ╚█████╗░
// ██║╚██╔╝██║ ██╔══██║ ██║░░██╗ ██╔══██╗ ██║░░██║ ░╚═══██╗
// ██║░╚═╝░██║ ██║░░██║ ╚█████╔╝ ██║░░██║ ╚█████╔╝ ██████╔╝
// ╚═╝░░░░░╚═╝ ╚═╝░░╚═╝ ░╚════╝░ ╚═╝░░╚═╝ ░╚════╝░ ╚═════╝░
//
// The staged interpreter. Everything in this file runs at compile time; none of
// it appears in the generated code. A constraint *type* goes in, an expression
// checking it comes out, and the compile-time environment that ties constraint
// variables together (`GenCtx`) is erased entirely -- there is no
// `ConstraintContext` and no dictionary at run time.
//
// There is no set of constraints this file knows about. It holds the shared
// environment and helpers, and dispatches every question to the constraint's
// own `ConstraintGen`; `EqAttr` and a constraint from a downstream dialect
// travel exactly the same path.

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||   COMPILE-TIME ENVIRONMENT  ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

/** A bound constraint variable: where to read it back from, and -- when the
  * binding could not be settled at compile time -- how to write it.
  *
  * With no `assign`, the variable costs nothing: it *is* the expression that
  * bound it, e.g. `adtOp.lhs.typ`, so a later use compiles to a bare comparison
  * against that path. With one, it is a local slot in the generated method,
  * needed when the binding happens somewhere its value cannot be named later:
  * inside a [[Param]] pattern, scoped to the match, or inside an [[AnyOf]]
  * alternative, which may not run at all. Still no allocation and no map -- the
  * JIT keeps a slot in a register.
  */
final case class Binding(
    read: Expr[Attribute],
    assign: Option[Expr[Attribute] => Expr[Unit]] = None,
)

/** The compile-time constraint-variable environment for one operation.
  *
  * Mutated as the interpreter walks an operation's constraints in order, so
  * that the first occurrence of a `Var` binds and later ones check. This is the
  * compile-time counterpart of xDSL's `ConstraintContext`, and unlike it, it
  * never exists at run time.
  *
  * @param owner
  *   the symbol owning the method being generated, needed to mint slots.
  */
final class GenCtx(using q: Quotes)(owner: q.reflect.Symbol):

  import q.reflect.*

  private val bindings = mutable.Map.empty[String, Binding]
  private val slots = mutable.ListBuffer.empty[(Symbol, Binding)]
  private var nameable = true

  def lookup(name: String): Option[Binding] = bindings.get(name)

  /** Run `body` in a position whose values cannot be named further down the
    * generated method, so anything binding inside it needs a slot.
    *
    * Entered by constraints that recurse under a pattern or a conditional.
    * Nesting is what makes this correct for, say, an [[AnyOf]] inside a
    * [[Param]]: once unnameable, always unnameable for that subtree.
    */
  def unnameable[A](body: => A): A =
    val saved = nameable
    nameable = false
    try body
    finally nameable = saved

  /** Bind `name` to `attr`, minting a slot if the current position needs one.
    *
    * The returned binding's `assign` is defined exactly when the caller must
    * emit a write rather than rely on the compile-time expression.
    */
  def bind(name: String, attr: Expr[Attribute]): Binding =
    val binding =
      if nameable then Binding(attr)
      else
        val sym = Symbol.newVal(
          owner,
          s"constraintVar_$name",
          TypeRepr.of[Attribute],
          Flags.Mutable,
          Symbol.noSymbol,
        )
        val b = Binding(
          Ref(sym).asExprOf[Attribute],
          Some((e: Expr[Attribute]) =>
            Assign(Ref(sym), e.asTerm).asExprOf[Unit]
          ),
        )
        slots += ((sym, b))
        b
    bindings(name) = binding
    binding

  /** Run `body`, and report which slots it minted.
    *
    * Lets [[AnyOf]] roll back exactly the variables its alternatives could have
    * bound, rather than every variable in the operation.
    */
  def tracking[A](body: => A): (A, Seq[Binding]) =
    val before = slots.length
    val result = body
    (result, slots.drop(before).map(_._2).toSeq)

  /** Declare every slot minted so far around `body`. */
  def declareSlots[A: Type](body: Expr[A]): Expr[A] =
    if slots.isEmpty then body
    else
      Block(
        slots.map((sym, _) => ValDef(sym, Some(Literal(NullConstant()))))
          .toList,
        body.asTerm,
      ).asExprOf[A]

/** Declare the slots the environment minted around `body`. */
def withDeclaredCells[A: Type](body: Expr[A])(using Quotes, GenCtx): Expr[A] =
  summon[GenCtx].declareSlots(body)

/** A custom failure message in scope, set by [[Msg]].
  *
  * Deliberately without a default given: a generator that forgets to thread it
  * should fail to compile rather than silently drop a [[Msg]] override.
  */
final case class ErrCtx(override_ : Option[String] = None)

/*≡==--==≡≡≡≡≡≡==--=≡≡*\
||      HELPERS       ||
\*≡==---==≡≡≡≡==---==≡*/

/** Select a member of an expression.
  *
  * @param obj
  *   the object to select the member from.
  * @param name
  *   the name of the member to select.
  */
def selectMember[T: Type](obj: Expr[?], name: String)(using
    Quotes
): Expr[T] =
  import quotes.reflect.*
  Select.unique(obj.asTerm, name).asExprOf[T]

/** The element types of a tuple type, as a list. */
def tupleTypes[T <: Tuple: Type](using Quotes): List[Type[?]] =
  Type.of[T] match
    case '[t *: ts]    => Type.of[t] :: tupleTypes[ts]
    case '[EmptyTuple] => Nil
    case _             =>
      quotes.reflect.report
        .errorAndAbort(
          s"Expected a fully known tuple of constraints, got ${Type.show[T]}"
        )

/** Build the failure expression for a check at `path` that expected `expected`,
  * honouring any [[Msg]] override in scope.
  */
def fail(path: String, expected: Expr[String], got: Expr[Attribute])(using
    Quotes,
    ErrCtx,
): Expr[OK[Unit]] =
  val prefix = Expr(summon[ErrCtx].override_ match
    case None      => s"$path: Expected "
    case Some(msg) =>
      s"$path: $msg\nUnderlying verification failure: Expected ")
  '{ Err($prefix + $expected + ", got " + $got.toString) }

/** Sequence checks, short-circuiting on the first failure.
  *
  * Emitted as `val r = a; if r.isError then r else b` rather than
  * `a.flatMap(_ => b)` so that the result never depends on the inliner.
  */
def chain(checks: Seq[Expr[OK[Unit]]])(using Quotes): Option[Expr[OK[Unit]]] =
  checks match
    case Seq()  => None
    case Seq(c) => Some(c)
    case cs     =>
      Some(
        cs.reduceRight((a, b) => '{ val r = $a; if r.isError then r else $b })
      )

/*≡==--==≡≡≡≡≡≡≡≡==--=≡≡*\
||      DISPATCHERS     ||
\*≡==---==≡≡≡≡≡≡==---==≡*/
//
// The whole interpreter. It recognises no constraint by name: it finds the
// constraint's `ConstraintGen` and asks it. `EqAttr` and a constraint defined
// in a downstream dialect are reached by exactly the same path.

/** Find `c`'s generator, or explain what is missing. */
private def genOrAbort(using Quotes)(c: Type[?]): ConstraintGen =
  import quotes.reflect.*
  val sym = TypeRepr.of(using c).dealias match
    case AppliedType(tycon, _) => tycon.typeSymbol
    case tpe                   => tpe.typeSymbol
  loadGen(sym.companionModule).getOrElse(
    quotes.reflect.report.errorAndAbort(
      s"${Type.show(using c)} has no ConstraintGen on its companion object, " +
        s"so it cannot be compiled.\nEither express it by composing existing " +
        s"constraints, or give its companion object a ConstraintGen (which " +
        s"must live in a module compiled before this one)."
    )
  )

/** Compile `c` into an expression checking that `attr` satisfies it.
  *
  * `None` means the constraint needs no code at all.
  */
def verifyC(c: Type[?], attr: Expr[Attribute], path: String)(using
    Quotes,
    GenCtx,
    ErrCtx,
): Option[Expr[OK[Unit]]] = genOrAbort(c).verify(c, attr, path)

/** Bind `e` to a local for the duration of `body`.
  *
  * Splicing an `Expr` twice duplicates its whole tree, not a reference to it,
  * so an attribute reached through a chain like `adtOp.lhs.typ` would be
  * recomputed once per constraint leaf that mentions it. Naming it once keeps
  * the generated code proportional to the constraint rather than to the number
  * of times it reads its subject.
  */
def letAttr(e: Expr[Attribute])(
    body: Expr[Attribute] => Option[Expr[OK[Unit]]]
)(using Quotes): Option[Expr[OK[Unit]]] =
  import quotes.reflect.*
  var built: Option[Expr[OK[Unit]]] = None
  val term = ValDef.let(Symbol.spliceOwner, "attr", e.asTerm) { ref =>
    built = body(ref.asExprOf[Attribute])
    built.getOrElse('{ OK() }).asTerm
  }
  // Drop the binding entirely when the constraint needed no code.
  built.map(_ => term.asExprOf[OK[Unit]])

/** Compile `c` into an expression producing the single attribute it admits, if
  * it admits exactly one.
  */
def inferC(c: Type[?])(using Quotes, GenCtx): Option[Expr[Attribute]] =
  genOrAbort(c).infer(c)

/** Bind `c`'s variables from a value already known to satisfy it. */
def bindC(c: Type[?], attr: Expr[Attribute])(using Quotes, GenCtx): Unit =
  genOrAbort(c).bind(c, attr)

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||   SINGLETON ATTRIBUTE REFERENCE  ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

/** Recover the value denoted by a singleton attribute type.
  *
  * This is the bridge that lets `EqAttr[f32.type]` name a concrete attribute:
  * `f32.type` is a `TermRef`, and the reference to its symbol is exactly the
  * expression `f32`.
  */
def attributeOf[To <: Attribute: Type](using Quotes): Expr[To] =
  import quotes.reflect.*
  TypeRepr.of[To].simplified match
    case tr: TermRef => Ref(tr.termSymbol).asExprOf[To]
    case t           =>
      report.errorAndAbort(
        s"EqAttr[${Type.show[To]}] needs a singleton type of a stable value, " +
          s"e.g. `EqAttr[f32.type]` for some `val f32 = Float32Type()`; " +
          s"got ${t.show}."
      )

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||   COMPILE-TIME GENERATOR LOADING    ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

/** The classloader the compiler uses to run macro implementations. It carries
  * the macro classpath, i.e. the dependencies of the module being compiled, so
  * any object defined in such a dependency is reachable from here and can be
  * called during this expansion.
  *
  * Note this is *not* `getClass.getClassLoader`: under a build tool that one is
  * the loader of the compiler itself, which cannot see the user's classpath.
  */
private def macroClassLoader: ClassLoader =
  Thread.currentThread().getContextClassLoader match
    case null   => getClass.getClassLoader
    case loader => loader

/** The generator for constraint type `c`: the object its companion denotes.
  *
  * A generator has to be a *value* callable during expansion, so implicit
  * search cannot serve -- it yields an `Expr`, not the object. Reflecting into
  * the macro classpath is the only mechanism, and the constraint's companion is
  * the least ceremony that gives a unique, discoverable answer.
  */
private def loadGen(using Quotes)(sym: quotes.reflect.Symbol) =
  import quotes.reflect.*
  if !sym.exists || !sym.flags.is(Flags.Module) then None
  else
    // A module symbol's `fullName` renders the runtime class name minus its
    // trailing '$'; nested objects are already '$'-separated.
    val binary =
      if sym.fullName.endsWith("$") then sym.fullName else s"${sym.fullName}$$"
    try
      Class.forName(binary, true, macroClassLoader).getField("MODULE$")
        .get(null) match
        case gen: ConstraintGen => Some(gen)
        case _                  => None
    catch case _: ReflectiveOperationException => None

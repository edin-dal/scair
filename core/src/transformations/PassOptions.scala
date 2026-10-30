package scair.transformations

import fastparse.*
import fastparse.NoWhitespace.*
import scair.clair.macros.defaultExprs
import scair.parse.explain

import java.util.Locale
import scala.compiletime.constValueTuple
import scala.compiletime.erasedValue
import scala.compiletime.summonAll
import scala.compiletime.summonInline
import scala.deriving.Mirror
import scala.quoted.*

/** `ConvertArithToLLVM` -> `convert-arith-to-llvm`, `maxIterations` ->
  * `max-iterations`.
  */
def kebabCase(s: String): String =
  // CLI names must not depend on the compiler or runtime locale.
  s.replaceAll("([A-Z]+)([A-Z][a-z])", "$1-$2")
    .replaceAll("([a-z0-9])([A-Z])", "$1-$2").toLowerCase(Locale.ROOT)

/** How to parse a pass option value of type T.
  *
  * An option is given on the command line as `key=v1,v2,...`. A bare `key` is
  * read as `key=true` for Boolean options.
  */
trait PassOption[T]:
  def typeName: String
  def parse[$: P]: P[T]

object PassOption:

  /** Simple enums, spelled as their kebab-cased case names.
    */
  inline given enumeration[E <: scala.reflect.Enum](using
      m: Mirror.SumOf[E]
  ): PassOption[E] =
    val names = constValueTuple[m.MirroredElemLabels].toList
      .map(l => kebabCase(l.toString))
    val values = summonAll[Tuple.Map[m.MirroredElemTypes, ValueOf]].toList
      .map(_.asInstanceOf[ValueOf[E]].value)
    PassOption.choice(names.zip(values))

  def scalar[T](
      name: String,
      read: String => Option[T],
  ): PassOption[T] = new:
    def typeName = name
    def parse[$: P]: P[T] =
      passOptionValueP.flatMap(v => read(v).fold(Fail)(Pass(_)))

  def choice[E](cases: Seq[(String, E)]): PassOption[E] =
    scalar(
      cases.map(_._1).mkString("|"),
      v => cases.collectFirst { case (`v`, e) => e },
    )

  given PassOption[Int] = scalar("Int", _.toIntOption)
  given PassOption[Long] = scalar("Long", _.toLongOption)
  given PassOption[Double] = scalar("Double", _.toDoubleOption)
  given PassOption[Boolean] = scalar("Boolean", _.toBooleanOption)
  given PassOption[String] = scalar("String", Some(_))

  given [T](using o: PassOption[T]): PassOption[Option[T]] = new:
    def typeName = s"Option[${o.typeName}]"
    def parse[$: P]: P[Option[T]] = o.parse.map(Some(_))

  given [T](using o: PassOption[T]): PassOption[Seq[T]] = new:
    def typeName = s"Seq[${o.typeName}]"
    def parse[$: P]: P[Seq[T]] = o.parse.rep(1, sep = ","./)

  given [T](using o: PassOption[Seq[T]]): PassOption[List[T]] = new:
    def typeName = o.typeName.replaceFirst("Seq", "List")
    def parse[$: P]: P[List[T]] = o.parse.map(_.toList)

/** Overrides the pass name derived from the class name. */
trait PassName[name <: String]

final case class PassField(
    name: String,
    option: PassOption[?],
    flag: Boolean,
    default: Option[() => Any],
)

/** A pass companion: its name and parser, derived from the pass case class.
  *
  * {{{
  * final case class MyPass(maxIterations: Int = 10) extends ModulePass
  *     derives PassCompanion
  * }}}
  * is run as `my-pass{max-iterations=3}`.
  */
final class PassCompanion[T <: ModulePass & Product](
    val name: String,
    val fields: Seq[PassField],
    build: Product => T,
):

  def parse[$: P]: P[T] =
    val seen = scala.collection.mutable.Set.empty[String]

    def option: P[(String, Any)] = passOptionNameP./.flatMap(key =>
      if !seen.add(key) then Fail(s"$name: duplicate option '$key'")
      else
        fields.find(_.name == key) match
          case None =>
            Fail(
              s"$name: unknown option '$key', expected one of: " +
                fields.map(_.name).mkString(", ")
            )
          case Some(field) =>
            val expected =
              s"$name: option '$key': expected ${field.option.typeName}"
            // A boolean key without `=` is read as `key=true`.
            ("=" ~/ field.option.parse.explain(expected) |
              (if field.flag then Pass(true) else Fail).explain(expected))
              .map(value => key -> value)
    )

    P(("{" ~/ " ".rep ~ option.rep(sep = " ".rep(1)) ~ " ".rep ~ "}").?)
      .flatMap(options =>
        val values = options.getOrElse(Seq.empty).toMap
        fields.find(f => !values.contains(f.name) && f.default.isEmpty) match
          case Some(field) =>
            Fail(s"$name: missing required option '${field.name}'")
          case None =>
            val arguments = fields
              .map(f => values.getOrElse(f.name, f.default.get.apply())).toArray
            Pass(build(Tuple.fromArray(arguments)))
      )

object PassCompanion:

  inline def derived[P <: ModulePass & Product](using
      m: Mirror.ProductOf[P]
  ): PassCompanion[P] =
    val labels = constValueTuple[m.MirroredElemLabels].toList
    val options = summonAll[Tuple.Map[m.MirroredElemTypes, PassOption]].toList
    new PassCompanion(
      passNameOf[P],
      labels.lazyZip(options).lazyZip(flagsOf[m.MirroredElemTypes])
        .lazyZip(defaultsOf[P]).map((l, o, f, d) =>
          PassField(kebabCase(l.toString), o.asInstanceOf[PassOption[?]], f, d)
        ),
      m.fromProduct,
    )

  inline def flagsOf[Ts <: Tuple]: List[Boolean] =
    inline erasedValue[Ts] match
      case _: EmptyTuple     => Nil
      case _: (head *: tail) =>
        (inline erasedValue[head] match
          case _: Boolean => true
          case _          => false
        ) :: flagsOf[tail]

  inline def passNameOf[P]: String = ${ passNameImpl[P] }

  inline def defaultsOf[P]: Seq[Option[() => Any]] = ${
    defaultsImpl[P]
  }

  def passNameImpl[P: Type](using Quotes): Expr[String] =
    import quotes.reflect.*
    Type.of[P] match
      case '[PassName[name]] =>
        Type.valueOfConstant[name] match
          case Some(name: String) => Expr(name)
          case _                  =>
            report.errorAndAbort("PassName requires a literal string type")
      case _ => Expr(kebabCase(TypeRepr.of[P].typeSymbol.name))

  def defaultsImpl[P: Type](using
      Quotes
  ): Expr[Seq[Option[() => Any]]] =
    import quotes.reflect.*
    val defaults = defaultExprs[P]
    Expr.ofSeq(
      TypeRepr.of[P].typeSymbol.caseFields.map(f =>
        defaults.get(f.name) match
          case Some(d) => '{ Some(() => $d) }
          case None    => '{ None }
      )
    )

/** Summons the companions of a tuple of pass types, in registration order. */
inline def summonPasses[Passes <: Tuple]: Seq[PassCompanion[?]] =
  inline erasedValue[Passes] match
    case _: EmptyTuple     => Seq()
    case _: (head *: tail) =>
      summonInline[PassCompanion[head & ModulePass & Product]] +:
        summonPasses[tail]

package scair.transformations

import scair.clair.macros.defaultExprs

import scala.annotation.StaticAnnotation
import scala.compiletime.constValueTuple
import scala.compiletime.summonAll
import scala.deriving.Mirror
import scala.quoted.*

/** `ConvertArithToLLVM` -> `convert-arith-to-llvm`, `maxIterations` ->
  * `max-iterations`.
  */
def kebabCase(s: String): String =
  s.replaceAll("([A-Z]+)([A-Z][a-z])", "$1-$2")
    .replaceAll("([a-z0-9])([A-Z])", "$1-$2").toLowerCase

/** How to read and print a pass option value of type T.
  *
  * An option is given on the command line as `key=v1,v2,...`, or as a bare
  * `key` with no values.
  */
trait PassOption[T]:
  def typeName: String
  def parse(values: Seq[String]): Either[String, T]
  def show(t: T): Seq[String]

trait LowPriorityPassOptions:

  /** Simple enums (or sealed traits of case objects), spelled as their
    * kebab-cased case names.
    */
  inline given enumeration[E](using m: Mirror.SumOf[E]): PassOption[E] =
    val names = constValueTuple[m.MirroredElemLabels].toList
      .map(l => kebabCase(l.toString))
    val values = summonAll[Tuple.Map[m.MirroredElemTypes, ValueOf]].toList
      .map(_.asInstanceOf[ValueOf[E]].value)
    PassOption.choice(names.zip(values))

object PassOption extends LowPriorityPassOptions:

  def scalar[T](
      name: String,
      read: String => Option[T],
      write: T => String = (_: T).toString,
  ): PassOption[T] = new:
    def typeName = name
    def parse(values: Seq[String]) = values match
      case Seq(v) => read(v).toRight(s"expected $name, got '$v'")
      case _      => Left(s"expected a single $name")
    def show(t: T) = Seq(write(t))

  def choice[E](cases: Seq[(String, E)]): PassOption[E] =
    scalar(
      cases.map(_._1).mkString("|"),
      v => cases.collectFirst { case (`v`, e) => e },
      e => cases.collectFirst { case (n, `e`) => n }.get,
    )

  given PassOption[Int] = scalar("Int", _.toIntOption)
  given PassOption[Long] = scalar("Long", _.toLongOption)
  given PassOption[Double] = scalar("Double", _.toDoubleOption)
  given PassOption[String] = scalar("String", Some(_))

  /** A bare `flag` means `flag=true`. */
  given PassOption[Boolean] = new:
    private val bool = scalar("Boolean", _.toBooleanOption)
    def typeName = bool.typeName
    def parse(values: Seq[String]) =
      if values.isEmpty then Right(true) else bool.parse(values)
    def show(b: Boolean) = bool.show(b)

  /** A bare `key` means `None`. */
  given [T](using o: PassOption[T]): PassOption[Option[T]] = new:
    def typeName = s"Option[${o.typeName}]"
    def parse(values: Seq[String]) =
      if values.isEmpty then Right(None) else o.parse(values).map(Some(_))
    def show(t: Option[T]) = t.fold(Seq())(o.show)

  given [T](using o: PassOption[T]): PassOption[Seq[T]] = new:
    def typeName = s"Seq[${o.typeName}]"
    def parse(values: Seq[String]) =
      values.foldLeft[Either[String, Seq[T]]](Right(Vector())) { (acc, v) =>
        acc.flatMap(ts => o.parse(Seq(v)).map(ts :+ _))
      }
    def show(ts: Seq[T]) = ts.flatMap(o.show)

  given [T](using o: PassOption[Seq[T]]): PassOption[List[T]] = new:
    def typeName = o.typeName.replaceFirst("Seq", "List")
    def parse(values: Seq[String]) = o.parse(values).map(_.toList)
    def show(ts: List[T]) = o.show(ts)

/** Overrides the pass name derived from the class name. */
final class passName(val name: String) extends StaticAnnotation

final case class PassField(
    name: String,
    option: PassOption[Any],
    default: Option[() => Any],
)

/** A pass definition: its name and options, derived from the pass case class.
  *
  * {{{
  * final case class MyPass(maxIterations: Int = 10) extends ModulePass
  *     derives PassDefs
  * }}}
  * is run as `my-pass{max-iterations=3}`.
  */
final class PassDefs[P <: ModulePass & Product](
    val name: String,
    val fields: Seq[PassField],
    build: Product => P,
):

  def fromArgs(args: Map[String, Seq[String]]): Either[String, P] =
    args.keys.find(k => !fields.exists(_.name == k)) match
      case Some(key) =>
        Left(
          s"$name: unknown option '$key', expected one of: " +
            fields.map(_.name).mkString(", ")
        )
      case None =>
        val values = fields.map(f =>
          args.get(f.name) match
            case Some(vs) =>
              f.option.parse(vs).left.map(e => s"$name: option '${f.name}': $e")
            case None =>
              f.default.map(d => d())
                .toRight(
                  s"$name: missing required option '${f.name}'"
                )
        )
        values.collectFirst { case Left(e) => e }
          .toLeft(build(Tuple.fromArray(values.map(_.toOption.get).toArray)))

  /** This pass in pipeline syntax, omitting options left at their default. */
  def show(pass: P): String =
    val options = fields.zip(pass.productIterator).collect {
      case (f, v) if !f.default.exists(_() == v) =>
        f.option.show(v) match
          case Seq()  => f.name
          case values => s"${f.name}=${values.map(quoted).mkString(",")}"
    }
    if options.isEmpty then name else options.mkString(s"$name{", " ", "}")

  /** e.g. `convert-arith-to-llvm{reconcile: Boolean = true}`; empty defaults
    * (`None`, empty lists) are left out.
    */
  def signature: String =
    if fields.isEmpty then name
    else
      fields.map(f =>
        s"${f.name}: ${f.option.typeName}" + f.default.fold(" (required)")(d =>
          f.option.show(d()) match
            case Seq()  => ""
            case values => " = " + values.map(quoted).mkString(",")
        )
      ).mkString(s"$name{", ", ", "}")

  private def quoted(value: String) =
    if value.matches("[A-Za-z0-9_.+-]+") then value
    else
      "\"" + value.replace("\\", "\\\\").replace("\"", "\\\"")
        .replace("\n", "\\n") + "\""

object PassDefs:

  def apply[P <: ModulePass & Product](using defs: PassDefs[P]) = defs

  inline def derived[P <: ModulePass & Product](using
      m: Mirror.ProductOf[P]
  ): PassDefs[P] =
    val labels = constValueTuple[m.MirroredElemLabels].toList
    val options = summonAll[Tuple.Map[m.MirroredElemTypes, PassOption]].toList
    new PassDefs(
      passNameOf[P],
      labels.lazyZip(options).lazyZip(defaultsOf[P]).map((l, o, d) =>
        PassField(kebabCase(l.toString), o.asInstanceOf[PassOption[Any]], d)
      ),
      m.fromProduct,
    )

  inline def passNameOf[P]: String = ${ passNameImpl[P] }

  inline def defaultsOf[P]: Seq[Option[() => Any]] = ${
    defaultsImpl[P]
  }

  def passNameImpl[P: Type](using Quotes): Expr[String] =
    import quotes.reflect.*
    val sym = TypeRepr.of[P].typeSymbol
    sym.getAnnotation(TypeRepr.of[passName].typeSymbol).map(_.asExpr) match
      case Some('{ new `passName`(${ Expr(name) }) }) => Expr(name)
      case Some(other)                                =>
        report.errorAndAbort(
          "@passName takes a literal string",
          other,
        )
      case None => Expr(kebabCase(sym.name))

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

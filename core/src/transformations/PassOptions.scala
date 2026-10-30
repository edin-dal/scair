package scair.transformations

import fastparse.*
import fastparse.NoWhitespace.*
import scair.clair.macros.defaultExprs
import scair.parse.explain

import scala.compiletime.constValue
import scala.compiletime.constValueTuple
import scala.compiletime.erasedValue
import scala.compiletime.error
import scala.compiletime.summonAll
import scala.compiletime.summonFrom
import scala.compiletime.summonInline
import scala.deriving.Mirror
import scala.quoted.*

/** The fixed pipeline value syntax; wrappers only contain scalar values. */
private[transformations] enum PassOptionParser:
  case Scalar(label: String, read: String => Option[Any], flag: Boolean = false)
  case Optional(value: Scalar)
  case Repeated(value: Scalar)

  def typeName: String = this match
    case Scalar(label, _, _) => label
    case Optional(value)     => s"Option[${value.typeName}]"
    case Repeated(value)     => s"Seq[${value.typeName}]"

  def isFlag: Boolean = this match
    case Scalar(_, _, flag) => flag
    case _                  => false

  def parse[$: P]: P[Any] = this match
    case Scalar(_, read, _) =>
      passOptionValueP.flatMap(v => read(v).fold(Fail)(Pass(_)))
    case Optional(value) => value.parse.map(Some(_))
    case Repeated(value) => value.parse.rep(1, sep = ","./)

/** Overrides the Scala class name used in pass pipelines. */
trait PassName[name <: String]

private[transformations] final case class PassField(
    name: String,
    option: PassOptionParser,
    default: Option[() => Any],
)

/** A pass companion: its name and parser, derived from the pass case class.
  *
  * {{{
  * final case class MyPass(maxIterations: Int = 10) extends ModulePass
  *     derives PassCompanion
  * }}}
  * is run as `MyPass{maxIterations=3}`.
  */
final class PassCompanion[T <: ModulePass & Product](
    val name: String,
    private[transformations] val fields: Seq[PassField],
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
              (if field.option.isFlag then Pass(true) else Fail)
                .explain(expected)).map(value => key -> value)
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
    val options = parsersOf[m.MirroredElemTypes]
    new PassCompanion(
      inline erasedValue[P] match
        case _: PassName[name] => constValue[name]
        case _                 => constValue[m.MirroredLabel],
      labels.lazyZip(options).lazyZip(defaultsOf[P])
        .map((l, o, d) => PassField(l.toString, o, d)),
      m.fromProduct,
    )

  private inline def parsersOf[Ts <: Tuple]: List[PassOptionParser] =
    inline erasedValue[Ts] match
      case _: EmptyTuple     => Nil
      case _: (head *: tail) => parserOf[head] :: parsersOf[tail]

  private inline def parserOf[T]: PassOptionParser =
    inline erasedValue[T] match
      case _: Option[a] =>
        summonFrom {
          case _: (T =:= Option[a]) => PassOptionParser.Optional(scalarOf[a])
          case _ => error("Pass optional fields must use Option[T]")
        }
      case _: Seq[a] =>
        summonFrom {
          case _: (T =:= Seq[a]) => PassOptionParser.Repeated(scalarOf[a])
          case _                 =>
            error(
              "Pass sequence fields must use Seq[T], not List or other collection types"
            )
        }
      case _ => scalarOf[T]

  private inline def scalarOf[T]: PassOptionParser.Scalar =
    import PassOptionParser.Scalar
    inline erasedValue[T] match
      case _: Int    => Scalar("Int", _.toIntOption)
      case _: Long   => Scalar("Long", _.toLongOption)
      case _: BigInt =>
        Scalar("BigInt", s => scala.util.Try(BigInt(s)).toOption)
      case _: Double  => Scalar("Double", _.toDoubleOption)
      case _: Boolean => Scalar("Boolean", _.toBooleanOption, flag = true)
      case _: String  => Scalar("String", Some(_))
      case _: scala.reflect.Enum =>
        enumOf[T](using summonInline[Mirror.SumOf[T]])
      case _ =>
        error(
          "Unsupported pass option type: expected Int, Long, BigInt, Double, Boolean, String, a simple enum, or Option/Seq of one of these (no nesting)"
        )

  private inline def enumOf[T](using
      m: Mirror.SumOf[T]
  ): PassOptionParser.Scalar =
    val names = constValueTuple[m.MirroredElemLabels].toList.map(_.toString)
    val values = summonAll[Tuple.Map[m.MirroredElemTypes, ValueOf]].toList
      .map(_.asInstanceOf[ValueOf[T]].value)
    PassOptionParser.Scalar(
      names.mkString("|"),
      s =>
        names.zip(values).collectFirst {
          case (name, value) if name == s => value
        },
    )

  inline def defaultsOf[P]: Seq[Option[() => Any]] = ${
    defaultsImpl[P]
  }

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

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

/** Overrides the Scala class name used in pass pipelines. */
trait PassName[name <: String]

/** Derived from a pass case class; its fields are the pipeline options. */
trait PassCompanion[T <: ModulePass & Product]:
  def name: String
  def parse[$: P]: P[T]

object PassCompanion:

  inline def derived[T <: ModulePass & Product](using
      m: Mirror.ProductOf[T]
  ): PassCompanion[T] =
    new PassCompanion[T]:
      val name = inline erasedValue[T] match
        case _: PassName[n] => constValue[n]
        case _              => constValue[m.MirroredLabel]

      def parse[$: P]: P[T] =
        val seen = scala.collection.mutable.Set.empty[String]
        def option: P[(String, Any)] = passOptionNameP./.flatMap(key =>
          if !seen.add(key) then Fail(s"$name: duplicate option '$key'")
          else if !constValueTuple[m.MirroredElemLabels].toList.contains(key)
          then
            Fail(
              s"$name: unknown option '$key', expected one of: " +
                constValueTuple[m.MirroredElemLabels].toList.mkString(", ")
            )
          else
            readOption[m.MirroredElemLabels, m.MirroredElemTypes](key, name)
              .map(value => key -> value)
        )
        P(("{" ~/ " ".rep ~ option.rep(sep = " ".rep(1)) ~ " ".rep ~ "}").?)
          .flatMap(options =>
            construct[T](options.getOrElse(Seq.empty).toMap) match
              case Left(field) =>
                Fail(s"$name: missing required option '$field'")
              case Right(pass) => Pass(pass)
          )

  private inline def readOption[Labels <: Tuple, Types <: Tuple](
      key: String,
      name: String,
  )(using P[Any]): P[Any] =
    inline erasedValue[(Labels, Types)] match
      case _: ((label *: labels), (t *: types)) =>
        if key == constValue[label] then
          val expected = s"$name: option '$key': expected ${typeName[t]}"
          ("=" ~/ value[t].explain(expected) |
            (inline erasedValue[t] match
              case _: Boolean => Pass(true)
              case _          => Fail
            ).explain(expected))
        else readOption[labels, types](key, name)
      case _: (EmptyTuple, EmptyTuple) =>
        Fail(s"$name: unknown option '$key'")

  private inline def typeName[T]: String = inline erasedValue[T] match
    case _: Option[a] => s"Option[${scalarName[a]}]"
    case _: Seq[a]    => s"Seq[${scalarName[a]}]"
    case _            => scalarName[T]

  private inline def scalarName[T]: String = inline erasedValue[T] match
    case _: scala.reflect.Enum =>
      enumNames[T](using summonInline[Mirror.SumOf[T]])
    case _: Int     => "Int"
    case _: Long    => "Long"
    case _: BigInt  => "BigInt"
    case _: Double  => "Double"
    case _: Boolean => "Boolean"
    case _: String  => "String"
    case _          =>
      error("Unsupported pass option type (nested wrappers are not supported)")

  private inline def value[T](using P[Any]): P[Any] =
    inline erasedValue[T] match
      case _: Option[a] =>
        summonFrom {
          case _: (T =:= Option[a]) => scalar[a].map(Some(_))
          case _ => error("Pass optional fields must use Option[T]")
        }
      case _: Seq[a] =>
        summonFrom {
          case _: (T =:= Seq[a]) => scalar[a].rep(1, sep = ","./)
          case _                 =>
            error(
              "Pass sequence fields must use Seq[T], not List or other collection types"
            )
        }
      case _ => scalar[T]

  private inline def scalar[T](using P[Any]): P[Any] =
    val read: String => Option[Any] = inline erasedValue[T] match
      case _: Int                => _.toIntOption
      case _: Long               => _.toLongOption
      case _: BigInt             => s => scala.util.Try(BigInt(s)).toOption
      case _: Double             => _.toDoubleOption
      case _: Boolean            => _.toBooleanOption
      case _: String             => Some(_)
      case _: scala.reflect.Enum =>
        enumReader[T](using summonInline[Mirror.SumOf[T]])
      case _ =>
        error(
          "Unsupported pass option type: expected Int, Long, BigInt, Double, Boolean, String, a simple enum, or Option/Seq of one of these (no nesting)"
        )
    passOptionValueP.flatMap(s => read(s).fold(Fail)(Pass(_)))

  private inline def enumNames[T](using m: Mirror.SumOf[T]): String =
    constValueTuple[m.MirroredElemLabels].toList.mkString("|")

  private inline def enumReader[T](using
      m: Mirror.SumOf[T]
  ): String => Option[Any] =
    val names = constValueTuple[m.MirroredElemLabels].toList.map(_.toString)
    val values = summonAll[Tuple.Map[m.MirroredElemTypes, ValueOf]].toList
      .map(_.asInstanceOf[ValueOf[T]].value)
    val cases = names.zip(values)
    s => cases.collectFirst { case (name, value) if name == s => value }

  private inline def construct[T](values: Map[String, Any])(using
      m: Mirror.ProductOf[T]
  ): Either[String, T] =
    ${ constructImpl[T]('values, 'm) }

  private def constructImpl[T: Type](
      values: Expr[Map[String, Any]],
      m: Expr[Mirror.ProductOf[T]],
  )(using Quotes): Expr[Either[String, T]] =
    import quotes.reflect.*
    val fields = TypeRepr.of[T].typeSymbol.caseFields
    val defaults = defaultExprs[T]
    val required = Expr
      .ofSeq(
        fields.filterNot(f => defaults.contains(f.name)).map(f => Expr(f.name))
      )
    val arguments = Expr.ofSeq(fields.map(f =>
      val key = Expr(f.name)
      defaults.get(f.name) match
        case Some(default) => '{ $values.getOrElse($key, $default) }
        case None          => '{ $values($key) }
    ))
    '{
      $required.find(field => ! $values.contains(field)) match
        case Some(field) => Left(field)
        case None => Right($m.fromProduct(Tuple.fromArray($arguments.toArray)))
    }

/** Summons the companions of a tuple of pass types, in registration order. */
inline def summonPasses[Passes <: Tuple]: Seq[PassCompanion[?]] =
  inline erasedValue[Passes] match
    case _: EmptyTuple     => Seq()
    case _: (head *: tail) =>
      summonInline[PassCompanion[head & ModulePass & Product]] +:
        summonPasses[tail]

def passOptionNameP[$: P]: P[String] = P(CharsWhileIn("A-Za-z0-9_.+\\-").!)

def passOptionValueP[$: P]: P[String] = P(
  ("\"" ~~ (CharsWhile(c => c != '"' && c != '\\').! | ("\\" ~~ AnyChar.!)).repX
    .map(_.mkString) ~~ "\"") | CharsWhileIn("A-Za-z0-9_./+\\-").!
)

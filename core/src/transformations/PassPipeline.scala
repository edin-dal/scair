package scair.transformations

import fastparse.*
import fastparse.NoWhitespace.*

/** A pass name and its raw option values, as written in a pipeline. */
final case class PassSpec(name: String, args: Map[String, Seq[String]])

/** Parses pass pipelines, following xDSL's syntax:
  * {{{
  * pipeline ::= element (`,` element)*
  * element  ::= name (`{` option (` ` option)* `}`)?
  * option   ::= key (`=` value (`,` value)*)?
  * value    ::= [A-Za-z0-9_.+-]+ | `"` string `"`
  * }}}
  */
object PassPipeline:

  private def ident[$: P] = P(CharsWhileIn("A-Za-z0-9_.+\\-").!)

  private def string[$: P] = P(
    "\"" ~
      (CharsWhile(c => c != '"' && c != '\\').! | ("\\" ~ AnyChar.!).map {
        case "n" => "\n"
        case "t" => "\t"
        case c   => c
      }).rep.map(_.mkString) ~ "\""
  )

  private def option[$: P] = P(
    ident ~ ("=" ~/ (string | ident).rep(1, sep = ",")).?
      .map(_.getOrElse(Seq()))
  )

  private def options[$: P] = P(
    "{" ~/ " ".rep ~ option.rep(sep = " ".rep(1)) ~ " ".rep ~ "}"
  ).map(_.toMap)

  private def element[$: P] = P(
    ident ~ options.?.map(_.getOrElse(Map()))
  ).map(PassSpec(_, _))

  private def pipeline[$: P] = P(
    Start ~ " ".rep ~ element.rep(sep = " ".rep ~ "," ~ " ".rep) ~ " ".rep ~ End
  )

  def parse(spec: String): Either[String, Seq[PassSpec]] =
    fastparse.parse(spec, pipeline(using _)) match
      case Parsed.Success(specs, _) => Right(specs)
      case failure: Parsed.Failure  =>
        Left(s"invalid pass pipeline '$spec': ${failure.trace().msg}")

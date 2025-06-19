package scair.clair.macros

import scala.quoted.*

import fastparse.*
import fastparse.SingleLineWhitespace.given

import scair.ir.*
import scair.Printer

trait Directive {
    def print(op : Expr[?], p: Expr[Printer])(using Quotes) : Expr[Unit]
}

case class LiteralDirective(
    literal: String
) extends Directive {
    def print(op : Expr[?], p: Expr[Printer])(using Quotes) : Expr[Unit] = {
        '{ $p.print(${Expr(literal)}) }
    }
}

case class AssemblyFormatDirective(
    directives: Seq[Directive]
) extends Directive {
    def print(op : Expr[?], p: Expr[Printer])(using Quotes) : Expr[Unit] = {
        Expr.block('{$p.print($op.asInstanceOf[Operation].name)} +: directives.map(_.print(op, p)).toList, '{})
    }
}

def assemblyFormat[$: P]: P[AssemblyFormatDirective] =
    directive.rep(1).map(AssemblyFormatDirective.apply)

def directive[$: P]: P[Directive] =
    literalDirective

def literalDirective[$: P]: P[LiteralDirective] =
    ("`" ~~ CharsWhile(_ != '`').! ~~ "`").map(LiteralDirective.apply)

def parseAssemblyFormat(format: String) : AssemblyFormatDirective = {
  parse(format, (x: fastparse.P[?]) => assemblyFormat(using x)) match
            case Parsed.Success(value, _) =>
              value
            case failure: Parsed.Failure =>
              throw new Exception(
                s"Failed to parse assembly format: ${failure.extra.trace().msg}"
              )
}
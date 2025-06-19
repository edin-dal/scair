package scair.clair.macros

import scala.quoted.*

import fastparse.*
import fastparse.SingleLineWhitespace.given

import scair.ir.*
import scair.Printer
import scair.clair.codegen.*

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

case class VariableDirective(
    construct: OpInputDef
) extends Directive {
    def print(op: Expr[?], p: Expr[Printer])(using Quotes) : Expr[Unit] = {
        construct match {
            case OperandDef(name = n, variadicity = v) =>
                v match {
                    case Variadicity.Variadic =>
                        '{ $p.printList(${selectMember(op, n).asExprOf[Seq[Operand[?]]]})(using 0) }
                }
        }
    }
}

case class TypeDirective(
    construct: OperandDef | ResultDef
) extends Directive {
    def print(op: Expr[?], p: Expr[Printer])(using Quotes) : Expr[Unit] = {
        construct match {
            case OperandDef(name = n, variadicity = v) =>
                v match
                    case Variadicity.Variadic =>
                        '{ $p.printList(${selectMember(op, n).asExprOf[Seq[Operand[?]]]}.map(_.typ))(using 0) }
        }
    }
}

case class AssemblyFormatDirective(
    directives: Seq[Directive]
) extends Directive {
    def print(op : Expr[?], p: Expr[Printer])(using Quotes) : Expr[Unit] = {
        Expr.block('{$p.print($op.asInstanceOf[Operation].name)} +: directives.map(_.print(op, p)).toList, '{})
    }
}


// Parses an assembly format identifier
// Those should match Scala's identifier rules, for maximum compatibility with the ADT
// fields. This is an approximation
def assemblyId[$: P]: P[String] =
    CharsWhileIn("a-zA-Z0-9_").!

def assemblyFormat[$: P](using opDef : OperationDef): P[AssemblyFormatDirective] =
    directive.rep(1).map(AssemblyFormatDirective.apply)

def directive[$: P](using opDef : OperationDef): P[Directive] =
    typeDirective | literalDirective | variableDirective

def literalDirective[$: P]: P[LiteralDirective] =
    ("`" ~~ CharsWhile(_ != '`').! ~~ "`").map(LiteralDirective.apply)

def variableDirective[$: P](using opDef : OperationDef) =
    ("$" ~~ assemblyId).map(
        name => 
          opDef.allDefs.find(_.name == name)
    ).filter(_.nonEmpty).map(_.get).map(VariableDirective(_))

def typeDirective[$: P](using opDef : OperationDef) =
    ("type(" ~~ variableDirective ~~ ")").map(d =>
        d match
        case VariableDirective(c : OperandDef) => Some(TypeDirective(c))
        case VariableDirective(c : ResultDef) => Some(TypeDirective(c))
        case _ => None
        ).filter(_.nonEmpty).map(_.get)

def parseAssemblyFormat(format: String, opDef : OperationDef) : AssemblyFormatDirective = {
  given OperationDef = opDef
  parse(format, (x: fastparse.P[?]) => assemblyFormat(using x)) match
            case Parsed.Success(value, index) =>
              value
            case failure: Parsed.Failure =>
              throw new Exception(
                s"Failed to parse assembly format: ${failure.extra.trace().msg}"
              )
}
package scair.clair.macros

import fastparse.*
import fastparse.SingleLineWhitespace.given
import fastparse.internal.MacroInlineImpls.*
import scair.Parser
import scair.Printer
import scair.clair.codegen.*
import scair.ir.*

import scala.quoted.*

trait Directive {
    def print(op : Expr[?], p: Expr[Printer])(using Quotes) : Expr[Unit]

    def parse(p: Expr[Parser])(using ctx: Expr[P[Any]])(using quotes: Quotes) : Expr[P[Any]]

}

case class LiteralDirective(
    literal: String
) extends Directive {
    def print(op : Expr[?], p: Expr[Printer])(using Quotes) : Expr[Unit] = {
        '{ $p.print(${Expr(literal)}) }
    }

    def parse(p: Expr[Parser])(using ctx: Expr[P[Any]])(using quotes: Quotes) : Expr[P[Unit]]= 
        literalStrMacro(Expr(literal))(ctx)
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

    def parse(p: Expr[Parser])(using ctx: Expr[P[Any]])(using quotes: Quotes) = 
        construct match {
            case OperandDef(name = n, variadicity = v) =>
                v match {
                    case Variadicity.Variadic =>
                        '{Parser.ValueUseList(using $ctx)}
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

    def parse(p: Expr[Parser])(using ctx: Expr[P[Any]])(using quotes: Quotes) = 
        construct match {
            case OperandDef(name = n, variadicity = v) =>
                v match {
                    case Variadicity.Variadic =>
                        '{
                            $p.AttributeValueList(using $ctx)
                        }
                }
        }
}

case class AssemblyFormatDirective(
    directives: Seq[Directive]
) extends Directive {
    def print(op : Expr[?], p: Expr[Printer])(using Quotes) : Expr[Unit] = {
        Expr.block('{$p.print($op.asInstanceOf[Operation].name)} +: directives.map(_.print(op, p)).toList, '{})
    }
    
    def parse(p: Expr[Parser])(using ctx: Expr[P[Any]])(using quotes: Quotes): Expr[P[Tuple]] = 
        
        val seq = directives.map(_.parse(p)(using ctx)).reduce((run, next) =>
            // This match to specialize the parsers types for fastparse's summoned Sequencer
            // cf fastparse.Implicits.Sequencer.
            //
            // I feel like there might be a more elegant way, but I spent enough time to
            // search for it and this sounds liek a localized enough compromise.
            (run, next) match
                case ('{$run: P[Unit]}, '{$next: P[n]}) =>
                    '{
                        given P[Any] = $ctx
                        $run ~ $next
                    }
                case ('{$run: P[r]}, '{$next: P[Unit]}) =>
                    '{
                        given P[Any] = $ctx
                        $run ~ $next
                    }
                case ('{$run: P[Tuple]}, '{$next: P[Any]}) =>
                    '{
                        given P[Any] = $ctx
                        given fastparse.Implicits.Sequencer[Tuple, Any, Tuple] = fastparse.Implicits.Sequencer.NarySequencer[Tuple, Any, Tuple](_ :* _)
                        `~`($run)[Any, Tuple]($next)
                    }
                case _ =>
                    '{
                        given P[Any] = $ctx
                        $run ~ $next
                    }
        )
        seq match
            case '{$tuple : P[Tuple]} =>
                tuple
            case '{$default : P[d]} =>
                println(s"Default case in assembly format parsing: ${Type.show[d]}")
                '{$default.map(Tuple1(_))}

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
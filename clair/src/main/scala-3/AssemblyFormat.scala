package scair.clair.macros

import fastparse.*
import fastparse.SingleLineWhitespace.given
import fastparse.internal.MacroInlineImpls.*
import scair.Parser
import scair.Printer
import scair.clair.codegen.*
import scair.ir.*

import scala.quoted.*

private inline def isalpha(c: Char): Boolean =
  (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')

private def printSpace(p: Expr[Printer], state: PrintingState)(using Quotes) =

  val print =
    if (state.shouldEmitSpace || !state.lastWasPunctuation)
      '{ $p.print(" ") }
    else '{}

  state.lastWasPunctuation = false;
  state.shouldEmitSpace = true;

  print

trait Directive {

  def print(op: Expr[?], p: Expr[Printer])(using
      state: PrintingState
  )(using Quotes): Expr[Unit]

  def parse(p: Expr[Parser])(using
      ctx: Expr[P[Any]]
  )(using quotes: Quotes): Expr[P[Any]]

}

case class LiteralDirective(
    literal: String
) extends Directive {

  private inline def shouldEmitSpaceBefore(
      inline lastWasPunctuation: Boolean
  ): Boolean =
    if (literal.size != 1 && literal != "->")
      true
    else if (lastWasPunctuation)
      !">)}],".contains(literal.head)
    else
      !"<>(){}[],".contains(literal.head)

  def print(op: Expr[?], p: Expr[Printer])(using
      state: PrintingState
  )(using Quotes): Expr[Unit] = {
    val toPrint =
      if (
        state.shouldEmitSpace && shouldEmitSpaceBefore(state.lastWasPunctuation)
      )
        " " + literal
      else
        literal

    state.shouldEmitSpace = literal.size != 1 || !"<({[".contains(literal.head)
    state.lastWasPunctuation = literal.head != '_' && !isalpha(literal.head)
    
    '{ $p.print(${ Expr(toPrint) }) }
  }

  def parse(p: Expr[Parser])(using
      ctx: Expr[P[Any]]
  )(using quotes: Quotes): Expr[P[Unit]] =
    literalStrMacro(Expr(literal))(ctx)

}

case class AttrDictDirective() extends Directive {

  def print(op: Expr[?], p: Expr[Printer])(using
      state: PrintingState
  )(using Quotes): Expr[Unit] = {
    state.lastWasPunctuation = false
    '{
      $p.printOptionalAttrDict(${
        selectMember(op, "attributes").asExprOf[DictType[String, Attribute]]
      }.toMap)(using 0)
    }
  }

  def parse(p: Expr[Parser])(using
      ctx: Expr[P[Any]]
  )(using quotes: Quotes): Expr[P[Map[String, Attribute]]] =
    '{ $p.OptionalAttributes(using $ctx) }

}

case class VariableDirective(
    construct: OpInputDef
) extends Directive {

  def print(op: Expr[?], p: Expr[Printer])(using
      state: PrintingState
  )(using Quotes): Expr[Unit] = {
    val space = printSpace(p, state)
    val printVar = construct match {
      case OperandDef(name = n, variadicity = v) =>
        v match {
          case Variadicity.Single =>
            '{ $p.print(${ selectMember(op, n).asExprOf[Operand[?]] }) }
          case Variadicity.Variadic =>
            '{
              $p.printList(${ selectMember(op, n).asExprOf[Seq[Operand[?]]] })(
                using 0
              )
            }
        }
    }
    Expr.block(List(space), printVar)
  }

  def parse(p: Expr[Parser])(using ctx: Expr[P[Any]])(using quotes: Quotes) =
    construct match {
      case OperandDef(name = n, variadicity = v) =>
        v match {
          case Variadicity.Single =>
            '{ Parser.ValueUse(using $ctx) }
          case Variadicity.Variadic =>
            '{ Parser.ValueUseList(using $ctx) }
        }
    }

}

case class TypeDirective(
    construct: OperandDef | ResultDef
) extends Directive {

  def print(op: Expr[?], p: Expr[Printer])(using
      state: PrintingState
  )(using Quotes): Expr[Unit] = {

    val space = printSpace(p, state)

    val printType = construct match {
      case OperandDef(name = n, variadicity = v) =>
        v match
          case Variadicity.Single =>
            '{ $p.print(${ selectMember(op, n).asExprOf[Operand[?]] }.typ) }
          case Variadicity.Variadic =>
            '{
              $p.printList(${
                selectMember(op, n).asExprOf[Seq[Operand[?]]]
              }.map(_.typ))(using 0)
            }
    }
    Expr.block(List(space), printType)
  }

  def parse(p: Expr[Parser])(using ctx: Expr[P[Any]])(using quotes: Quotes) =
    construct match {
      case OperandDef(name = n, variadicity = v) =>
        v match {
          case Variadicity.Single =>
            '{ $p.AttributeValue(using $ctx) }
          case Variadicity.Variadic =>
            '{ $p.AttributeValueList(using $ctx) }
        }
    }

}

def chainParsers(
    run: Expr[P[Any]],
    next: Expr[P[Any]]
)(using quotes: Quotes, ctx: Expr[P[Any]]): Expr[P[Any]] =

  // This match to specialize the parsers types for fastparse's summoned Sequencer
  // cf fastparse.Implicits.Sequencer.
  //
  // I feel like there might be a more elegant way, but I spent enough time to
  // search for it and this sounds liek a localized enough compromise.
  (run, next) match
    case ('{ $run: P[Unit] }, '{ $next: P[n] }) =>
      '{
        given P[Any] = $ctx
        $run ~ $next
      }
    case ('{ $run: P[r] }, '{ $next: P[Unit] }) =>
      '{
        given P[Any] = $ctx
        $run ~ $next
      }
    case ('{ $run: P[Tuple] }, '{ $next: P[Any] }) =>
      '{
        given P[Any] = $ctx
        given fastparse.Implicits.Sequencer[Tuple, Any, Tuple] =
          fastparse.Implicits.Sequencer
            .NarySequencer[Tuple, Any, Tuple](_ :* _)
        `~`($run)[Any, Tuple]($next)
      }
    case _ =>
      '{
        given P[Any] = $ctx
        $run ~ $next
      }

case class PrintingState(
    var shouldEmitSpace: Boolean = true,
    var lastWasPunctuation: Boolean = false
)

case class AssemblyFormatDirective(
    directives: Seq[Directive]
) {

  def print(op: Expr[?], p: Expr[Printer])(using Quotes): Expr[Unit] = {
    given PrintingState = PrintingState()
    Expr.block(
      '{ $p.print($op.asInstanceOf[Operation].name) } +: directives
        .map(_.print(op, p))
        .toList,
      '{}
    )
  }

  def parse(p: Expr[Parser])(using
      ctx: Expr[P[Any]]
  )(using quotes: Quotes): Expr[P[Tuple]] =

    directives.map(_.parse(p)).reduce(chainParsers) match
      case '{ $tuple: P[Tuple] } =>
        tuple
      case '{ $default: P[d] } =>
        '{ $default.map(Tuple1(_)) }

  def parsedDirectives: Seq[Directive] =
    directives.filter(_ match
      case _: LiteralDirective => false
      case _                   => true)

  def extractGenerationArgs(
      opDef: OperationDef,
      p: Expr[Parser],
      parsed: Expr[Tuple]
  )(using Quotes) =
    import quotes.reflect.report

    val operandNames = Map
      .from(
        parsedDirectives.zipWithIndex.flatMap((d, i) =>
          d match
            case VariableDirective(OperandDef(name = name)) => Some(name -> i)
            case _                                          => None
        )
      )
      .mapValues(i => '{ $parsed.productIterator.toList(${ Expr(i) }) })
    val operandNamesArg =
      Expr.ofList(opDef.operands.map(od => (operandNames(od.name))))
    val flatNames = '{
      $operandNamesArg.flatMap(op =>
        op match
          case op: String      => Seq(op)
          case op: Seq[String] => op
      )
    }

    val operandTypes = Map
      .from(
        parsedDirectives.zipWithIndex.flatMap((d, i) =>
          d match
            case TypeDirective(OperandDef(name = name)) => Some(name -> i)
            case _                                      => None
        )
      )
      .mapValues(i => '{ $parsed.productIterator.toList(${ Expr(i) }) })
    val operandTypesArg =
      Expr.ofList(opDef.operands.map(od => (operandTypes(od.name))))
    val flatTypes = '{
      $operandTypesArg.flatMap(op =>
        op match
          case op: Attribute      => Seq(op)
          case op: Seq[Attribute] => op
      )
    }

    val attrDictIndex = parsedDirectives.zipWithIndex.find(_._1 match
      case _: AttrDictDirective => true
      case _                    => false) match
      case Some((AttrDictDirective(), i)) => i
      case None                           =>
        report.errorAndAbort(
          "Assembly format directive must contain an `attr-dict` directive"
        )

    val attrDict = '{
      $parsed(${ Expr(attrDictIndex) }).asInstanceOf[Map[String, Attribute]]
    }

    '{
      $p.generateOperation(
        opName = ${ Expr(opDef.name) },
        operandsNames = $flatNames,
        operandsTypes = $flatTypes,
        attributes = $attrDict
      )
    }

}

// Parses an assembly format identifier
// Those should match Scala's identifier rules, for maximum compatibility with the ADT
// fields. This is an approximation
transparent inline def assemblyId[$: P]: P[String] =
  CharsWhileIn("a-zA-Z0-9_").!

transparent inline def assemblyFormat[$: P](using
    opDef: OperationDef
): P[AssemblyFormatDirective] =
  directive.rep(1).map(AssemblyFormatDirective.apply)

transparent inline def directive[$: P](using
    opDef: OperationDef
): P[Directive] =
  typeDirective | literalDirective | variableDirective | attrDictDirective

transparent inline def literalDirective[$: P]: P[LiteralDirective] =
  ("`" ~~ CharsWhile(_ != '`').! ~~ "`").map(LiteralDirective.apply)

transparent inline def variableDirective[$: P](using opDef: OperationDef) =
  ("$" ~~ assemblyId)
    .map(name => opDef.allDefs.find(_.name == name))
    .filter(_.nonEmpty)
    .map(_.get)
    .map(VariableDirective(_))

transparent inline def typeDirective[$: P](using opDef: OperationDef) =
  ("type(" ~~ variableDirective ~~ ")")
    .map(d =>
      d match
        case VariableDirective(c: OperandDef) => Some(TypeDirective(c))
        case VariableDirective(c: ResultDef)  => Some(TypeDirective(c))
        case _                                => None
    )
    .filter(_.nonEmpty)
    .map(_.get)

transparent inline def attrDictDirective[$: P]: P[AttrDictDirective] =
  ("attr-dict").map(_ => AttrDictDirective())

def parseAssemblyFormat(
    format: String,
    opDef: OperationDef
): AssemblyFormatDirective = {
  given OperationDef = opDef
  parse(format, (x: fastparse.P[?]) => assemblyFormat(using x)) match
    case Parsed.Success(value, index) =>
      value
    case failure: Parsed.Failure =>
      throw new Exception(
        s"Failed to parse assembly format: ${failure.extra.trace().msg}"
      )
}

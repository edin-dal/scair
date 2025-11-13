package scair.clair.macros

import fastparse.*
import fastparse.SingleLineWhitespace.given
import fastparse.internal.MacroInlineImpls.*
import scair.Parser
import scair.Printer
import scair.clair.codegen.*
import scair.ir.*
import scair.macros.*

import scala.quoted.*

/** Utility function to check if a character is alphabetic */
private inline def isalpha(c: Char): Boolean =
  (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')

/** Prints a space in the output if required by printing context. Manages
  * spacing rules based on punctuation and previous tokens.
  */
private def printSpace(p: Expr[Printer], state: PrintingState)(using Quotes) =

  val print =
    if state.shouldEmitSpace || !state.lastWasPunctuation then
      '{ $p.print(" ") }
    else '{}

  state.lastWasPunctuation = false;
  state.shouldEmitSpace = true;

  print

/** Base trait for assembly format directives. Directives represent a unit of
  * the assembly format, and can generate printing and parsing implementations
  * for their constructs.
  */
trait Directive:

  /** Generate a specialized printer for the directive's subset of an
    * operation's definition.
    * @param op
    *   The Operation argument of the generated printer.
    * @param p
    *   The Printer argument of the generated printer.
    * @param state
    *   The current printer generation state.
    * @return
    *   Specialized code to print one construct of a specific operation type.
    */
  def print(op: Expr[?], p: Expr[Printer])(using
      state: PrintingState
  )(using Quotes): Expr[Unit]

  /** Generate a specialized parser for the directive's subset of an operation's
    * definition.
    * @param p
    *   The Parser argument of the generated parser.
    * @return
    *   Specialized code to parse one construct of a specific operation type.
    */
  def parse(p: Expr[Parser])(using
      ctx: Expr[P[Any]]
  )(using quotes: Quotes): Expr[P[Any]]

  // Did the directive parse something?
  def parsed(p: Expr[?])(using Quotes): Expr[Boolean] =
    import quotes.reflect.*
    report.errorAndAbort(
      s"Directive $this is not supposed to be used as an optional group's leading element, or is not implemented yet."
    )

  def isPresent(op: Expr[?])(using Quotes): Expr[Boolean] =
    import quotes.reflect.*
    report.errorAndAbort(
      s"Directive $this is not supposed to be used as an anchor, or is not implemented yet."
    )

/** Directive for literal text in the assembly format. Examples include
  * keywords, punctuation, and other fixed strings. Typically used to clarify
  * semantic or solve ambiguity.
  */
case class LiteralDirective(
    literal: String
) extends Directive:

  private inline def shouldEmitSpaceBefore(
      inline lastWasPunctuation: Boolean
  ): Boolean =
    if literal.size != 1 && literal != "->" then true
    else if lastWasPunctuation then !">)}],".contains(literal.head)
    else !"<>(){}[],".contains(literal.head)

  def print(op: Expr[?], p: Expr[Printer])(using
      state: PrintingState
  )(using Quotes): Expr[Unit] =
    val toPrint =
      if state.shouldEmitSpace && shouldEmitSpaceBefore(
          state.lastWasPunctuation
        )
      then " " + literal
      else literal

    state.shouldEmitSpace = literal.size != 1 || !"<({[".contains(literal.head)
    state.lastWasPunctuation = literal.head != '_' && !isalpha(literal.head)

    '{ $p.print(${ Expr(toPrint) }) }

  def parse(p: Expr[Parser])(using
      ctx: Expr[P[Any]]
  )(using quotes: Quotes): Expr[P[Unit]] =
    literalStrMacro(Expr(literal))(ctx)

/** Directive for an operation's attribute dictionnary. Its presence is
  * mandatory in every declarative assembly format, as this ensures the
  * operation's unknown added attributes are carried by its syntax.
  */
case class AttrDictDirective() extends Directive:

  def print(op: Expr[?], p: Expr[Printer])(using
      state: PrintingState
  )(using Quotes): Expr[Unit] =
    state.lastWasPunctuation = false
    '{
      $p.printOptionalAttrDict(${
        op.member[DictType[String, Attribute]]("attributes")
      }.toMap)(using 0)
    }

  def parse(p: Expr[Parser])(using
      ctx: Expr[P[Any]]
  )(using quotes: Quotes): Expr[P[Map[String, Attribute]]] =
    '{ $p.OptionalAttributes(using $ctx) }

/** Directive for variables, handling operations' individual constructs
  * (operands, results, regions, successors or properties).
  */
case class VariableDirective(
    construct: OpInputDef
) extends Directive:

  def print(op: Expr[?], p: Expr[Printer])(using
      state: PrintingState
  )(using Quotes): Expr[Unit] =
    val space = printSpace(p, state)
    val printVar = construct match
      case OperandDef(name = n, variadicity = v) =>
        v match
          case Variadicity.Single =>
            '{ $p.print(${ op.member[Operand[Attribute]](n) }) }
          case Variadicity.Variadic =>
            '{
              $p.printList(${ op.member[Seq[Operand[Attribute]]](n) })(using
                0
              )
            }
          case Variadicity.Optional =>
            '{
              $p.printList(${
                op.member[Option[Operand[Attribute]]](n)
              })(using 0)
            }
      case ResultDef(name = n, variadicity = v) =>
        v match
          case Variadicity.Single =>
            '{ $p.print(${ op.member[Result[Attribute]](n) }) }
          case Variadicity.Variadic =>
            '{
              $p.printList(${ op.member[Seq[Result[Attribute]]](n) })(using
                0
              )
            }
          case Variadicity.Optional =>
            '{
              $p.printList(${ op.member[Option[Result[Attribute]]](n) })(using
                0
              )
            }
      case OpPropertyDef(name = n, variadicity = v) =>
        v match
          case Variadicity.Single =>
            '{
              $p.print(${ op.member[Attribute](n) })
            }
          case Variadicity.Optional =>
            '{
              ${ op.member[Option[Attribute]](n) }.foreach($p.print)
            }

    Expr.block(List(space), printVar)

  def parse(p: Expr[Parser])(using ctx: Expr[P[Any]])(using quotes: Quotes) =
    construct match
      case OperandDef(name = n, variadicity = v) =>
        v match
          case Variadicity.Single =>
            '{ Parser.ValueUse(using $ctx) }
          case Variadicity.Variadic =>
            '{ Parser.ValueUseList(using $ctx) }
          case Variadicity.Optional =>
            '{ given P[?] = $ctx; Parser.ValueUse.? }
      case OpPropertyDef(name = n, variadicity = v) =>
        v match
          case Variadicity.Single =>
            '{
              $p.Attribute(using $ctx)
            }
          case Variadicity.Optional =>
            '{
              given P[?] = $ctx; $p.Attribute.?
            }

  override def parsed(p: Expr[?])(using Quotes) =
    import quotes.reflect.*
    construct match
      case MayVariadicOpInputDef(
            name = n,
            variadicity = Variadicity.Variadic
          ) =>
        '{ ${ p }.asInstanceOf[Seq[?]].size > 0 }
      case MayVariadicOpInputDef(
            name = n,
            variadicity = Variadicity.Optional
          ) =>
        '{ ${ p }.asInstanceOf[Option[?]].isDefined }
      case d: OpInputDef =>
        report.errorAndAbort(
          s"Variable directives can only be used as anchors when variadic or optional, tried to use `${d.name}`."
        )

  override def isPresent(op: Expr[?])(using Quotes) =
    construct match
      case OpInputDef(name = n) => parsed(op.member[Any](n))

/** Directive for types of individual operands or results.
  */
case class TypeDirective(
    construct: OperandDef | ResultDef
) extends Directive:

  def print(op: Expr[?], p: Expr[Printer])(using
      state: PrintingState
  )(using Quotes): Expr[Unit] =

    val space = printSpace(p, state)

    val printType = construct match
      case MayVariadicOpInputDef(name = n, variadicity = Variadicity.Single) =>
        '{ $p.print(${ op.member[Value[?]](n) }.typ) }
      case MayVariadicOpInputDef(
            name = n,
            variadicity = Variadicity.Variadic
          ) =>
        '{
          $p.printList(${ op.member[Seq[Value[?]]](n) }.map(_.typ))(using
            0
          )
        }
      case MayVariadicOpInputDef(
            name = n,
            variadicity = Variadicity.Optional
          ) =>
        '{
          ${ op.member[Option[Value[?]]](n) }
            .map(_.typ)
            .map($p.print)
            .getOrElse(())
        }

    Expr.block(List(space), printType)

  def parse(p: Expr[Parser])(using ctx: Expr[P[Any]])(using quotes: Quotes) =
    construct match
      case MayVariadicOpInputDef(name = n, variadicity = v) =>
        v match
          case Variadicity.Single =>
            '{ $p.Type(using $ctx) }
          case Variadicity.Variadic =>
            '{ $p.TypeList(using $ctx) }
          case Variadicity.Optional =>
            '{ given P[?] = $ctx; $p.Type.? }

  // Ew; but works
  override def parsed(p: Expr[?])(using Quotes): Expr[Boolean] =
    VariableDirective(construct).parsed(p)

  override def isPresent(op: Expr[?])(using Quotes): Expr[Boolean] =
    VariableDirective(construct).isPresent(op)

case class OptionalGroupDirective(
    anchor: Directive,
    directives: Seq[Directive]
) extends Directive:

  def parse(
      p: Expr[Parser]
  )(using ctx: Expr[P[Any]])(using quotes: Quotes): Expr[P[Tuple]] =
    '{
      given P[?] = $ctx
      ${ directives.head.parse(p) }.flatMap(v =>
        if ${ directives.head.parsed('v) } then
          ${ AssemblyFormatDirective(directives.tail).parseTuple(p) }
            .map(v *: _)
        else
          Pass(${
            Expr.ofTupleFromSeq(
              AssemblyFormatDirective(directives).parsedDirectives.map(empty)
            )
          })
      )
    }

  def print(op: Expr[?], p: Expr[Printer])(using
      state: PrintingState
  )(using Quotes) =
    '{
      if ${ anchor.isPresent(op) }
      then ${ Expr.block(directives.map(_.print(op, p)).toList, '{ () }) }
    }

  def empty(directive: Directive)(using quotes: Quotes) =
    import quotes.reflect.*
    directive match
      case VariableDirective(
            MayVariadicOpInputDef(variadicity = Variadicity.Variadic)
          ) | TypeDirective(
            MayVariadicOpInputDef(variadicity = Variadicity.Variadic)
          ) =>
        '{ Seq() }
      case VariableDirective(
            MayVariadicOpInputDef(variadicity = Variadicity.Optional)
          ) | TypeDirective(
            MayVariadicOpInputDef(variadicity = Variadicity.Optional)
          ) =>
        '{ None }
      case _ =>
        report.errorAndAbort(
          s"Unsupported directive in optional group: $directive"
        )

/** Helper function to chain parsers together using fastparse's sequencing
  * operator. Handles different return types by matching on the specific parser
  * types.
  */
transparent inline def chainParsers(
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
    case ('{ $run: P[Tuple] }, '{ $next: P[Tuple] }) =>
      '{
        given P[Any] = $ctx
        given fastparse.Implicits.Sequencer[Tuple, Tuple, Tuple] =
          fastparse.Implicits.Sequencer
            .NarySequencer[Tuple, Tuple, Tuple](_ ++ _)
        `~`($run)[Tuple, Tuple]($next)
      }
    case ('{ $run: P[Any] }, '{ $next: P[Tuple] }) =>
      '{
        given P[Any] = $ctx
        given fastparse.Implicits.Sequencer[Any, Tuple, Tuple] =
          fastparse.Implicits.Sequencer
            .NarySequencer[Any, Tuple, Tuple](_ *: _)
        `~`($run)[Tuple, Tuple]($next)
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

/** Holds state during printer generation to manage spacing and punctuation.
  */
case class PrintingState(
    var shouldEmitSpace: Boolean = true,
    var lastWasPunctuation: Boolean = false
)

/** Declarative assembly format representation. Contains a sequence of
  * directives that define the format.
  */
case class AssemblyFormatDirective(
    directives: Seq[Directive]
):

  def print(op: Expr[?], p: Expr[Printer])(using Quotes): Expr[Unit] =
    given PrintingState = PrintingState()
    Expr.block(
      '{ $p.print($op.asInstanceOf[Operation].name) } +: directives
        .map(_.print(op, p))
        .toList,
      '{}
    )

  /** Generates a parser for this assembly format. It currently simply chains
    * all the individual directives parsers. This will parse each directives'
    * output into a tuple, which can then be used to generate the operation.
    */
  def parseTuple(p: Expr[Parser])(using
      ctx: Expr[P[Any]]
  )(using quotes: Quotes): Expr[P[Tuple]] =

    directives.map(_.parse(p)).reduce(chainParsers) match
      case '{ $tuple: P[Tuple] } =>
        tuple
      case '{ $default: P[d] } =>
        '{ $default.map(Tuple1(_)) }

  /** The list of directives that parse into something, as opposed to literal.
    * Helps with indexing the parsed tuple.
    */
  def parsedDirectives: Seq[Directive] =
    directives
      .flatMap(
        _ match
          case OptionalGroupDirective(_, ds) => ds
          case d: Directive                  => Some(d)
      )
      .filter(_ match
        case _: LiteralDirective => false
        case _                   => true)

  /** Use the operation definition to generate logic to build the operation from
    * the parsed tuple.
    */
  def buildOperation(
      opDef: OperationDef,
      p: Expr[Parser],
      parsed: Expr[Tuple],
      resNames: Expr[Seq[String]]
  )(using Quotes) =
    import quotes.reflect.report

    // TODO: Bunch of refactoring to do here, akin to what happened in main Macros.
    // I'm postponing it for now, as I feel it is worth considering actual code reuse
    // with Macros, rather than just overengineering this part.
    val operandNames = Map
      .from(
        parsedDirectives.zipWithIndex.flatMap((d, i) =>
          d match
            case VariableDirective(OperandDef(name = name)) =>
              Some(name -> '{ $parsed(${ Expr(i) }) }.asExprOf[Any])
            case _ => None
        )
      )
    val operandNamesArg =
      Expr.ofList(opDef.operands.map(od => (operandNames(od.name))))
    val flatOperandNames = '{
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
            case TypeDirective(OperandDef(name = name)) =>
              Some(name -> '{ $parsed(${ Expr(i) }) }.asExprOf[Any])
            case _ => None
        )
      )
    val operandTypesArg =
      Expr.ofList(opDef.operands.map(od => (operandTypes(od.name))))
    val flatOperandTypes = '{
      $operandTypesArg.flatMap(op =>
        op match
          case op: Attribute      => Seq(op)
          case op: Seq[Attribute] => op
      )
    }

    val resultTypes = Map
      .from(
        parsedDirectives.zipWithIndex.flatMap((d, i) =>
          d match
            case TypeDirective(ResultDef(name = name)) =>
              Some(name -> '{ $parsed(${ Expr(i) }) }.asExprOf[Any])
            case _ => None
        )
      )
    val resultTypesArg =
      Expr.ofList(opDef.results.map(od => (resultTypes(od.name))))
    val flatResultTypes = '{
      $resultTypesArg.flatMap(op =>
        op match
          case op: Attribute      => Seq(op)
          case op: Seq[Attribute] => op
      )
    }

    val attrDictIndex = parsedDirectives.zipWithIndex.find(_._1 match
      case _: AttrDictDirective => true
      case _                    => false) match
      case Some((AttrDictDirective(), i)) => i
      case _                              =>
        report.errorAndAbort(
          "Assembly format directive must contain an `attr-dict` directive"
        )

    val attrDict = '{
      $parsed(${ Expr(attrDictIndex) }).asInstanceOf[Map[String, Attribute]]
    }

    val propertiesNames = Map
      .from(
        parsedDirectives.zipWithIndex.flatMap((d, i) =>
          d match
            case VariableDirective(OpPropertyDef(name = name)) =>
              Some(name -> i)
            case _ => None
        )
      )
      .mapValues((i) => '{ $parsed(${ Expr(i) }) }.asExprOf[Any])
      .map((n, e) => '{ (${ Expr(n) }, $e) })
      .toSeq
    val propertiesDict = '{ Map.from(${ Expr.ofList(propertiesNames) }) }
    // This pushes the constructor disptching to runtime just like with generic syntax.
    // TODO: This should at least generate a call to the right Unstructured[T] constructor.
    // Or of course, directly T if so we choose.
    '{
      $p.generateOperation(
        opName = ${ Expr(opDef.name) },
        operandsNames = $flatOperandNames,
        operandsTypes = $flatOperandTypes,
        resultsNames = $resNames,
        resultsTypes = $flatResultTypes,
        attributes = $attrDict,
        properties = $propertiesDict.asInstanceOf[Map[String, Attribute]]
      )
    }

  /** Generate a complete specialized parser for this assembly format and
    * operation definition. This will parse the assembly format into a tuple of
    * parsed values, which can then be used to build the operation using the
    * operation definition.
    *
    * @param opDef
    *   The OperationDef for which to generate the parser.
    * @param p
    *   The Parser argument of the generated parser.
    * @param ctx
    *   The P context for the generated parser.
    * @return
    *   Specialized code to parse an assembly format into an Operation.
    */
  def parse(opDef: OperationDef, p: Expr[Parser], resNames: Expr[Seq[String]])(
      using quotes: Quotes
  ) =
    '{ (ctx: P[Any]) ?=>
      ${ parseTuple(p)(using '{ ctx }) }.map(parsed =>
        ${ buildOperation(opDef, p, '{ parsed }, resNames) }
      )
    }

case class Anchor(directive: Directive)

/** Parses an assembly format identifier. Those should match Scala's identifier
  * rules, for maximum compatibility with the ADT fields; this is an
  * approximation.
  */
transparent inline def assemblyId[$: P]: P[String] =
  CharsWhileIn("a-zA-Z0-9_").!

/** Parser for the complete assembly format. Parses one or more directives into
  * an AssemblyFormatDirective.
  */
transparent inline def assemblyFormat[$: P](using
    opDef: OperationDef
): P[AssemblyFormatDirective] =
  (directive.rep(1) ~ End).map(AssemblyFormatDirective.apply)

/** Parser for any directive.
  */
transparent inline def directive[$: P](using
    opDef: OperationDef
): P[Directive] =
  typeDirective | literalDirective | variableDirective | attrDictDirective | optionalGroupDirective

/** Parser for literal directives. Parses text enclosed in backticks as a
  * literal directive.
  */
transparent inline def literalDirective[$: P]: P[LiteralDirective] =
  ("`" ~~ CharsWhile(_ != '`').! ~~ "`").map(LiteralDirective.apply)

/** Parser for variable directives. Parses a dollar sign followed by an
  * identifier, which references a construct of the Operation.
  */
transparent inline def variableDirective[$: P](using opDef: OperationDef) =
  ("$" ~~ assemblyId)
    .map(name => opDef.allDefs.find(_.name == name))
    .filter(_.nonEmpty)
    .map(_.get)
    .map(VariableDirective(_))

/** Parser for type directives. Parses "type($var)" where $var is a variable
  * directive.
  */
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

/** Parser for attribute dictionary directives. Parses the keyword "attr-dict"
  * into an AttrDictDirective.
  */
transparent inline def attrDictDirective[$: P]: P[AttrDictDirective] =
  ("attr-dict").map(_ => AttrDictDirective())

transparent inline def possiblyAnchoredDirective[$: P](using
    opDef: OperationDef
) =
  (directive ~~ "^").map(Anchor.apply) | directive

def optionalGroupDirective[$: P](using opDef: OperationDef): P[Directive] =
  ("(" ~ possiblyAnchoredDirective.rep(1) ~ ")" ~ "?")./.filter(
    _.count(_.isInstanceOf[Anchor]) == 1
  )
    .map(directives =>
      val anchor = directives
        .find(_.isInstanceOf[Anchor])
        .get
        .asInstanceOf[Anchor]
        .directive
      val flatDirectives = directives.map(d =>
        d match
          case Anchor(d)    => d
          case d: Directive => d
      )
      OptionalGroupDirective(anchor, flatDirectives)
    )

/** Parse a declarative assembly format string into an AssemblyFormatDirective,
  * its internal representation for implementation generation.
  */
def parseAssemblyFormat(
    format: String,
    opDef: OperationDef
): AssemblyFormatDirective =
  given OperationDef = opDef
  parse(format, (x: fastparse.P[?]) => assemblyFormat(using x)) match
    case Parsed.Success(value, index) =>
      value
    case failure: Parsed.Failure =>
      throw new Exception(
        s"Failed to parse assembly format: ${failure.extra.trace().msg}"
      )

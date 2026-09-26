package scair.clair.macros

import fastparse.*
import fastparse.SingleLineWhitespace.given
import fastparse.internal.MacroInlineImpls.*
import scair.dialects.builtin.UnitAttr
import scair.ir.*
import scair.parse.*
import scair.print.Printer

import scala.quoted.*
import scala.util.control.NonFatal

//
// ░█████╗░ ░██████╗ ░██████╗ ███████╗ ███╗░░░███╗ ██████╗░ ██╗░░░░░ ██╗░░░██╗
// ██╔══██╗ ██╔════╝ ██╔════╝ ██╔════╝ ████╗░████║ ██╔══██╗ ██║░░░░░ ╚██╗░██╔╝
// ███████║ ╚█████╗░ ╚█████╗░ █████╗░░ ██╔████╔██║ ██████╦╝ ██║░░░░░ ░╚████╔╝░
// ██╔══██║ ░╚═══██╗ ░╚═══██╗ ██╔══╝░░ ██║╚██╔╝██║ ██╔══██╗ ██║░░░░░ ░░╚██╔╝░░
// ██║░░██║ ██████╔╝ ██████╔╝ ███████╗ ██║░╚═╝░██║ ██████╦╝ ███████╗ ░░░██║░░░
// ╚═╝░░╚═╝ ╚═════╝░ ╚═════╝░ ╚══════╝ ╚═╝░░░░░╚═╝ ╚═════╝░ ╚══════╝ ░░░╚═╝░░░
//
// ███████╗ ░█████╗░ ██████╗░ ███╗░░░███╗ ░█████╗░ ████████╗
// ██╔════╝ ██╔══██╗ ██╔══██╗ ████╗░████║ ██╔══██╗ ╚══██╔══╝
// █████╗░░ ██║░░██║ ██████╔╝ ██╔████╔██║ ███████║ ░░░██║░░░
// ██╔══╝░░ ██║░░██║ ██╔══██╗ ██║╚██╔╝██║ ██╔══██║ ░░░██║░░░
// ██║░░░░░ ╚█████╔╝ ██║░░██║ ██║░╚═╝░██║ ██║░░██║ ░░░██║░░░
// ╚═╝░░░░░ ░╚════╝░ ╚═╝░░╚═╝ ╚═╝░░░░░╚═╝ ╚═╝░░╚═╝ ░░░╚═╝░░░
//

/*≡≡=---=≡≡≡≡≡≡=---=≡≡*\
||   REPRESENTATION   ||
\*≡==----=≡≡≡≡=----==≡*/

/** A directive of a declarative assembly format, i.e., a unit of its syntax. */
enum Directive:

  /** Literal text, e.g., keywords or punctuation, typically used to clarify
    * semantics or solve ambiguities.
    */
  case Literal(text: String)

  /** The operation's attribute dictionary. It also carries the properties the
    * rest of the format does not spell out, as MLIR does, so that no property
    * is lost by a custom syntax.
    */
  case AttrDict

  /** One of the operation's operands, regions or properties. */
  case Variable(construct: OperandDef | RegionDef | OpPropertyDef)

  /** The type(s) of one of the operation's operands or results. */
  case TypeOf(construct: OperandDef | ResultDef)

  /** Directives only present if the anchor's construct is. */
  case OptionalGroup(
      anchor: Directive.Variable | Directive.TypeOf,
      body: List[Directive],
  )

import Directive.*

/** The construct a variable or type directive refers to. */
private def constructOf(directive: Variable | TypeOf): MayVariadicOpInputDef =
  directive match
    case Variable(construct) => construct
    case TypeOf(construct)   => construct

/** A unit property carries no value of its own: its presence is the
  * information, and it is spelled out by the literals around it, as in
  * ``(`volatile` $volatile_^)?``. Such a variable prints nothing and parses as
  * present without consuming any input.
  */
private object Flag:

  def unapply(d: OpPropertyDef)(using Quotes): Boolean =
    d.tpe match
      case '[UnitAttr] => true
      case _           => false

/** A validated declarative assembly format.
  *
  * @param directives
  *   The format's directives.
  * @param unspelled
  *   The properties the format does not spell out, carried by `attr-dict`.
  */
final case class AssemblyFormatDef(
    directives: List[Directive],
    unspelled: List[OpPropertyDef],
):

  /** Generate a specialized printer for this format.
    *
    * @param opDef
    *   The definition of the operation.
    * @param op
    *   The ADT operation to print.
    * @param p
    *   The printer to print with.
    */
  def print[T: Type](opDef: OperationDef, op: Expr[T], p: Expr[Printer])(using
      Quotes
  ): Expr[Unit] =
    FormatPrinter(this, op, p).print(opDef.name)

  /** Generate a specialized parser for this format, directly constructing the
    * ADT operation.
    *
    * @param opDef
    *   The definition of the operation.
    * @param p
    *   The Parser argument of the generated parser.
    * @param resNames
    *   The names of the results, parsed before the operation's name.
    */
  def parse[T <: Operation: Type](
      opDef: OperationDef,
      p: Expr[Parser],
      resNames: Expr[Seq[String]],
  )(using Quotes): Expr[P[Any] ?=> P[T]] =
    '{ (ctx: P[Any]) ?=>
      ${ FormatParser[T](this, opDef, p, resNames, 'ctx).parse }
    }

/*≡≡=---=≡≡≡≡≡=---=≡≡*\
||      SYNTAX       ||
\*≡==----=≡≡≡=----==≡*/

/** A directive as written, before resolving its variables. */
private enum Syntax:
  case Literal(text: String)
  case AttrDict
  case Variable(name: String)
  case TypeOf(name: String)
  case OptionalGroup(elements: List[(Syntax, Boolean)])

/** An assembly format identifier. Those should match Scala's identifier rules,
  * for maximum compatibility with the ADT fields; this is an approximation.
  */
private def identifierP[$: P]: P[String] =
  CharsWhileIn("a-zA-Z0-9_").!

private def formatP[$: P]: P[List[Syntax]] = (syntaxP.rep(1) ~ End)
  .map(_.toList)

private def syntaxP[$: P]: P[Syntax] =
  typeOfP | literalP | variableP | attrDictP | optionalGroupP

private def literalP[$: P] = ("`" ~~ CharsWhile(_ != '`').! ~~ "`")
  .map(Syntax.Literal(_))

private def variableP[$: P] = ("$" ~~ identifierP).map(Syntax.Variable(_))

private def typeOfP[$: P] = ("type(" ~~ "$" ~~ identifierP ~~ ")")
  .map(Syntax.TypeOf(_))

private def attrDictP[$: P] =
  P("attr-dict").map(_ => Syntax.AttrDict)

private def optionalGroupP[$: P] =
  ("(" ~ (syntaxP ~~ "^".!.?.map(_.isDefined)).rep(1) ~ ")" ~ "?")./
    .map(elements => Syntax.OptionalGroup(elements.toList))

/** Parse a declarative assembly format string into its validated
  * representation.
  *
  * @param format
  *   The declarative assembly format.
  * @param opDef
  *   The definition of the operation it is the format of.
  */
def parseAssemblyFormat(format: String, opDef: OperationDef)(using
    Quotes
): AssemblyFormatDef =
  import quotes.reflect.report

  def abort(msg: String) =
    report.errorAndAbort(s"Invalid assembly format for ${opDef.name}: $msg")

  def lookup(name: String) =
    opDef.allDefs.find(_.name == name)
      .getOrElse(abort(s"`$$$name` is not a construct of the operation."))

  def resolve(syntax: Syntax): Directive = syntax match
    case Syntax.Literal(text)  => Literal(text)
    case Syntax.AttrDict       => AttrDict
    case Syntax.Variable(name) =>
      lookup(name) match
        case d: (OperandDef | RegionDef | OpPropertyDef) => Variable(d)
        case _: ResultDef                                =>
          abort(s"results can only be spelled as `type($$$name)`.")
        case _: SuccessorDef =>
          abort(s"successors are not supported, found `$$$name`.")
    case Syntax.TypeOf(name) =>
      lookup(name) match
        case d: (OperandDef | ResultDef) => TypeOf(d)
        case _ => abort(s"`$$$name` has no type, in `type($$$name)`.")
    case Syntax.OptionalGroup(elements) =>
      val body = elements.map((s, _) => resolve(s))
      val anchor = (body zip elements)
        .collect { case (d, (_, true)) => d } match
        case Seq(anchor: (Variable | TypeOf)) => anchor
        case _                                =>
          abort("an optional group needs exactly one `^` anchor.")
      if constructOf(anchor).variadicity == Variadicity.Single then
        abort(s"the anchor `${constructOf(anchor).name}` is not optional.")
      body.foreach {
        case _: Literal             =>
        case d: (Variable | TypeOf) =>
          if constructOf(d).variadicity == Variadicity.Single then
            abort(
              s"`${constructOf(d).name}` in an optional group is not optional."
            )
        case _ =>
          abort(
            "`attr-dict` and optional groups cannot be in an optional group."
          )
      }
      OptionalGroup(anchor, body)

  val directives = fastparse.parse(format, formatP(using _)) match
    case Parsed.Success(syntax, _) => syntax.map(resolve)
    case failure: Parsed.Failure   => abort(failure.trace().msg)

  val all = directives.flatMap {
    case OptionalGroup(_, body) => body
    case d                      => List(d)
  }
  def once(d: Directive, spelling: String) = all.count(_ == d) match
    case 1 =>
    case 0 => abort(s"$spelling is missing.")
    case _ => abort(s"$spelling is spelled more than once.")

  if opDef.successors.nonEmpty then abort("successors are not supported.")
  once(AttrDict, "`attr-dict`")
  opDef.operands.foreach(d =>
    once(Variable(d), s"`$$${d.name}`")
    once(TypeOf(d), s"`type($$${d.name})`")
  )
  opDef.results.foreach(d => once(TypeOf(d), s"`type($$${d.name})`"))
  opDef.regions.foreach(d => once(Variable(d), s"`$$${d.name}`"))
  opDef.properties.foreach(d =>
    if all.count(_ == Variable(d)) > 1 then
      abort(s"`$$${d.name}` is spelled more than once.")
  )

  AssemblyFormatDef(
    directives,
    unspelled = opDef.properties.filterNot(d => all.contains(Variable(d)))
      .toList,
  )

/*≡≡=---=≡≡≡≡≡=---=≡≡*\
||     PRINTING      ||
\*≡==----=≡≡≡=----==≡*/

/** The spacing context of a printed directive.
  *
  * @param emitSpace
  *   Whether the previous directive allows a space after it.
  * @param afterPunctuation
  *   Whether the previous directive is punctuation.
  */
private final case class Spacing(
    emitSpace: Boolean = true,
    afterPunctuation: Boolean = false,
):

  /** Whether a value is preceded by a space, and the spacing after it. */
  def value: (Boolean, Spacing) =
    (emitSpace || !afterPunctuation, Spacing())

  /** Whether a literal is preceded by a space, and the spacing after it. */
  def literal(text: String): (Boolean, Spacing) =
    val head = text.head
    val space = emitSpace &&
      (if text.size != 1 && text != "->" then true
       else if afterPunctuation then !">)}],".contains(head)
       else !"<>(){}[],".contains(head))
    val isAlpha = (head >= 'a' && head <= 'z') || (head >= 'A' && head <= 'Z')
    (
      space,
      Spacing(
        emitSpace = text.size != 1 || !"<({[".contains(head),
        afterPunctuation = head != '_' && !isAlpha,
      ),
    )

/** Printer generation for a format, specialized to an ADT operation.
  *
  * The spacing between directives is entirely decided at compile time, folding
  * a [[Spacing]] through them. An optional group's directives are folded as if
  * present.
  */
private class FormatPrinter[T: Type](
    format: AssemblyFormatDef,
    op: Expr[T],
    p: Expr[Printer],
)(using Quotes):

  def print(name: String): Expr[Unit] =
    Expr.block(
      '{ $p.print(${ Expr(name) }) } :: printAll(format.directives, Spacing())
        ._1,
      '{},
    )

  private def printAll(
      directives: List[Directive],
      spacing: Spacing,
  ): (List[Expr[Unit]], Spacing) =
    directives
      .foldLeft((List.empty[Expr[Unit]], spacing)) {
        case ((printed, spacing), directive) =>
          val (code, next) = printOne(directive, spacing)
          (printed ++ code, next)
      }

  private def printOne(
      directive: Directive,
      spacing: Spacing,
  ): (Option[Expr[Unit]], Spacing) =
    directive match
      case Literal(text) =>
        val (space, next) = spacing.literal(text)
        val printed = if space then " " + text else text
        (Some('{ $p.print(${ Expr(printed) }) }), next)
      case AttrDict =>
        (
          Some('{ $p.printOptionalAttrDict($attrDict) }),
          spacing.copy(afterPunctuation = false),
        )
      case Variable(Flag())       => (None, spacing)
      case d: (Variable | TypeOf) =>
        val (space, next) = spacing.value
        val value = printValue(d)
        (Some(if space then '{ $p.print(" "); $value } else value), next)
      case OptionalGroup(anchor, body) =>
        val (printed, next) = printAll(body, spacing)
        (
          Some('{
            if ${ isPresent(anchor) } then ${ Expr.block(printed, '{}) }
          }),
          next,
        )

  private def printValue(directive: Variable | TypeOf): Expr[Unit] =
    directive match
      case Variable(d: OperandDef) =>
        printEach[Value[Attribute]](d)(v => '{ $p.print($v) })
      case Variable(d: RegionDef) =>
        printEach[Region](d)(r => '{ $p.print($r) })
      case Variable(d: OpPropertyDef) =>
        printEach[Attribute](d)(a => '{ $p.print($a) })
      case TypeOf(d) =>
        printEach[Value[Attribute]](d)(v => '{ $p.print($v.typ) })

  /** Print each element of a construct, according to its variadicity. */
  private def printEach[E: Type](d: MayVariadicOpInputDef)(
      one: Expr[E] => Expr[Unit]
  ): Expr[Unit] =
    d.variadicity match
      case Variadicity.Single   => one(selectMember[E](op, d.name))
      case Variadicity.Optional =>
        '{ ${ selectMember[Option[E]](op, d.name) }.foreach(e => ${ one('e) }) }
      case Variadicity.Variadic =>
        '{
          $p.printListF(
            ${ selectMember[Seq[E]](op, d.name) },
            e => ${ one('e) },
          )
        }

  private def isPresent(anchor: Variable | TypeOf): Expr[Boolean] =
    val d = constructOf(anchor)
    d.variadicity match
      case Variadicity.Optional =>
        '{ ${ selectMember[Option[Any]](op, d.name) }.isDefined }
      case _ =>
        '{ ${ selectMember[Seq[Any]](op, d.name) }.nonEmpty }

  /** The attribute dictionary, along with the unspelled properties. */
  private def attrDict: Expr[Map[String, Attribute]] =
    format.unspelled.foldLeft(
      selectMember[Map[String, Attribute]](op, "attributes")
    )((dict, d) =>
      val name = Expr(d.name)
      d.variadicity match
        case Variadicity.Single =>
          '{ $dict.updated($name, ${ selectMember[Attribute](op, d.name) }) }
        case Variadicity.Optional =>
          '{
            val current = $dict
            ${ selectMember[Option[Attribute]](op, d.name) } match
              case Some(value) => current.updated($name, value)
              case None        => current
          }
    )

/*≡≡=---=≡≡≡≡≡=---=≡≡*\
||      PARSING      ||
\*≡==----=≡≡≡=----==≡*/

/** A parser for some directives capturing no value, e.g., literals. */
private final case class Skipped(parser: Expr[P[Unit]])

/** A parser for some directives capturing values.
  *
  * @param parser
  *   The parser, yielding a (nested pair of) captured value(s).
  * @param empty
  *   The value yielded when the directives are absent, if they can be.
  * @param bind
  *   Where each directive's value lands in the parser's result.
  */
private final case class Captured[T](
    parser: Expr[P[T]],
    empty: Option[Expr[T]],
    bind: Expr[T] => Map[Directive, Expr[Any]],
)(using val tpe: Type[T])

private type Parsed = Skipped | Captured[?]

/** A resolution step of a construct, from its captured values.
  *
  * @param name
  *   The name of the construct.
  * @param parser
  *   The parser yielding the resolved construct.
  */
private final case class Step[V](name: String, parser: Expr[P[V]])(using
    val tpe: Type[V]
)

/** Parser generation for a format, specialized to an ADT operation.
  *
  * Each directive's parser has a precise type; sequencing them with fastparse's
  * `~` yields nested pairs of captured values, which are then bound to the
  * operation's constructs to directly construct it.
  */
private class FormatParser[T <: Operation: Type](
    format: AssemblyFormatDef,
    opDef: OperationDef,
    p: Expr[Parser],
    resNames: Expr[Seq[String]],
    ctx: Expr[P[Any]],
)(using Quotes):

  def parse: Expr[P[T]] =
    sequence(format.directives) match
      case c: Captured[t] =>
        given Type[t] = c.tpe
        '{
          given P[Any] = $ctx
          ${ c.parser }.flatMap((parsed: t) => ${ build(c.bind('parsed)) })
        }
      // A format always captures its attribute dictionary.
      case Skipped(_) => quotes.reflect.report.errorAndAbort("Unreachable")

  private def sequence(directives: List[Directive]): Parsed =
    directives.map(parseOne).reduceLeft(andThen)

  private def andThen(first: Parsed, second: Parsed): Parsed =
    (first, second) match
      case (Skipped(a), Skipped(b)) =>
        Skipped('{ given P[Any] = $ctx; $a ~ $b })
      case (Skipped(a), b: Captured[t]) =>
        given Type[t] = b.tpe
        b.copy(parser = '{ given P[Any] = $ctx; $a ~ ${ b.parser } })
      case (a: Captured[t], Skipped(b)) =>
        given Type[t] = a.tpe
        a.copy(parser = '{ given P[Any] = $ctx; ${ a.parser } ~ $b })
      case (a: Captured[s], b: Captured[t]) =>
        given Type[s] = a.tpe
        given Type[t] = b.tpe
        Captured[(s, t)](
          '{ given P[Any] = $ctx; ${ a.parser } ~ ${ b.parser } },
          a.empty.zip(b.empty).map((x, y) => '{ ($x, $y) }),
          v => a.bind('{ $v._1 }) ++ b.bind('{ $v._2 }),
        )

  private def parseOne(directive: Directive): Parsed =
    directive match
      case Literal(text) =>
        Skipped(literalStrMacro(Expr(text))(ctx))
      case AttrDict =>
        captured(directive, '{ optionalAttributesP(using $ctx, $p) })
      case Variable(d @ Flag()) =>
        lifted(directive, d.variadicity, '{ Pass(UnitAttr())(using $ctx) })
      case Variable(d: OperandDef) =>
        lifted(
          directive,
          d.variadicity,
          '{ operandNameP(using $ctx) },
          '{ operandNamesP(using $ctx) },
        )
      case Variable(d: RegionDef) =>
        lifted(
          directive,
          d.variadicity,
          '{
            given P[Any] = $ctx
            given Parser = $p
            regionP()
          },
          '{
            given P[Any] = $ctx
            given Parser = $p
            regionP().rep(sep = ",")
          },
        )
      case Variable(d: OpPropertyDef) =>
        lifted(directive, d.variadicity, '{ attributeP(using $ctx, $p) })
      case TypeOf(d) =>
        lifted(
          directive,
          d.variadicity,
          '{ typeP(using $ctx, $p) },
          '{ typeListP(using $ctx, $p) },
        )
      case OptionalGroup(_, body) =>
        val first = parseOne(body.head) match
          case Skipped(parser) => parser
          case c: Captured[?]  => c.parser
        sequence(body) match
          case c: Captured[t] =>
            given Type[t] = c.tpe
            // Validated: every directive of a group can be absent.
            val empty = c.empty.get
            c.copy(parser = '{
              given P[Any] = $ctx
              (&($first) ~~ ${ c.parser }) | Pass($empty)
            })
          // Validated: a group captures its anchor.
          case Skipped(_) => quotes.reflect.report.errorAndAbort("Unreachable")

  private def captured[V: Type](
      directive: Directive,
      parser: Expr[P[V]],
      empty: Option[Expr[V]] = None,
  ): Captured[V] =
    Captured(parser, empty, v => Map(directive -> v))

  /** The parser of a construct, lifted from the parser of one element according
    * to its variadicity.
    */
  private def lifted[E: Type](
      directive: Directive,
      variadicity: Variadicity,
      one: Expr[P[E]],
      many: => Expr[P[Seq[E]]] = quotes.reflect.report
        .errorAndAbort("Unreachable"),
  ): Captured[?] =
    variadicity match
      case Variadicity.Single   => captured(directive, one)
      case Variadicity.Optional =>
        captured(
          directive,
          '{ given P[Any] = $ctx; $one.? },
          Some('{ None }),
        )
      case Variadicity.Variadic =>
        captured(directive, many, Some('{ Seq() }))

  /*≡≡=--=≡≡ CONSTRUCTION ≡≡=--=≡≡*/

  /** Construct the operation from the captured values.
    *
    * Operands and results are resolved, in that order, before directly calling
    * the ADT's constructor.
    */
  private def build(values: Map[Directive, Expr[Any]]): Expr[P[T]] =
    val attrDict = values(AttrDict).asExprOf[Map[String, Attribute]]
    val operands = opDef.operands
      .map(d => resolveOperand(d, values(Variable(d)), values(TypeOf(d))))
    val (results, expected) =
      resolveResults(opDef.results.map(d => d -> values(TypeOf(d))).toList)
    resolving(operands.toList) { operands =>
      checkResults(expected) {
        resolving(results) { results =>
          val regions = opDef.regions.map(d => d.name -> values(Variable(d)))
          val properties = opDef.properties.map(d =>
            val value = values.get(Variable(d)) match
              case None => '{ $attrDict.get(${ Expr(d.name) }) }
              case Some(value) if d.variadicity == Variadicity.Single =>
                '{ Some(${ value.asExprOf[Attribute] }) }
              case Some(value) => value.asExprOf[Option[Attribute]]
            d.name -> propertyArgument(d, value)
          )
          val attributes =
            if format.unspelled.isEmpty then attrDict
            else '{ $attrDict -- ${ Expr(format.unspelled.map(_.name)) } }
          val constructed = construct[T](
            opDef.operands.map(d => d.name -> operands(d.name)) ++
              opDef.results.map(d => d.name -> results(d.name)) ++ regions ++
              properties
          )
          '{
            given P[Any] = $ctx
            try
              val op = $constructed
              op.attributes ++= $attributes
              Pass(op)
            catch case NonFatal(e) => Fail(e.getMessage)
          }
        }
      }
    }

  /** Bind the values of resolution steps, in order, to build a parser. */
  private def resolving(
      steps: List[Step[?]],
      resolved: Map[String, Expr[Any]] = Map(),
  )(k: Map[String, Expr[Any]] => Expr[P[T]]): Expr[P[T]] =
    steps match
      case Nil                          => k(resolved)
      case (step: Step[v]) :: remaining =>
        given Type[v] = step.tpe
        '{
          ${ step.parser }.flatMapX((value: v) =>
            ${ resolving(remaining, resolved + (step.name -> 'value))(k) }
          )
        }

  /** Resolve a construct from its parsed name(s) and type(s), lifting the
    * resolution of one element according to its variadicity.
    */
  private def resolve[V: Type](
      d: MayVariadicOpInputDef,
      kind: String,
      names: Expr[Any],
      types: Expr[Any],
  )(one: (Expr[String], Expr[Attribute]) => Expr[P[V]]): Step[?] =
    d.variadicity match
      case Variadicity.Single =>
        Step(d.name, one(names.asExprOf[String], types.asExprOf[Attribute]))
      case Variadicity.Optional =>
        val name = names.asExprOf[Option[String]]
        val typ = types.asExprOf[Option[Attribute]]
        Step(
          d.name,
          '{
            FormatParsing.optional($name, $typ, ${ Expr(kind) })((n, t) =>
              ${ one('n, 't) }
            )(using $ctx)
          },
        )
      case Variadicity.Variadic =>
        val name = names.asExprOf[Seq[String]]
        val typ = types.asExprOf[Seq[Attribute]]
        Step(
          d.name,
          '{
            FormatParsing.variadic($name, $typ, ${ Expr(kind) })((n, t) =>
              ${ one('n, 't) }
            )(using $ctx)
          },
        )

  private def resolveOperand(
      d: OperandDef,
      names: Expr[Any],
      types: Expr[Any],
  ): Step[?] =
    d.tpe match
      case '[type t <: Attribute; `t`] =>
        resolve(d, "operand", names, types)((name, typ) =>
          '{ FormatParsing.operand[t]($name, $typ)(using $ctx, $p) }
        )

  /** Resolve results, in order, distributing the result names over them.
    *
    * @return
    *   The resolution steps, and the expected number of result names.
    */
  private def resolveResults(
      types: List[(ResultDef, Expr[Any])]
  ): (List[Step[?]], Expr[Int]) =
    val sizes = types.map((d, types) =>
      d.variadicity match
        case Variadicity.Single   => Expr(1)
        case Variadicity.Optional =>
          '{ ${ types.asExprOf[Option[Attribute]] }.size }
        case Variadicity.Variadic =>
          '{ ${ types.asExprOf[Seq[Attribute]] }.size }
    )
    val offsets = sizes
      .scanLeft(Expr(0))((offset, size) => '{ $offset + $size })
    val steps = types.lazyZip(sizes).lazyZip(offsets).map {
      case ((d, types), size, offset) =>
        val names = d.variadicity match
          case Variadicity.Single   => '{ $resNames($offset) }
          case Variadicity.Optional =>
            '{ Option.when($size == 1)($resNames($offset)) }
          case Variadicity.Variadic =>
            '{ $resNames.slice($offset, $offset + $size) }
        d.tpe match
          case '[type t <: Attribute; `t`] =>
            resolve(d, "result", names, types)((name, typ) =>
              '{ FormatParsing.result[t]($name, $typ)(using $ctx, $p) }
            )
    }
    (steps, offsets.last)

  /** Check the number of result names against the expected one. */
  private def checkResults(expected: Expr[Int])(
      k: => Expr[P[T]]
  ): Expr[P[T]] =
    '{
      given P[Any] = $ctx
      val count = $expected
      if $resNames.length != count then
        Fail(
          s"Number of results (${$resNames.length}) does not match the number of the corresponding result types ($count) in \"${${
              Expr(opDef.name)
            }}\"."
        )
      else $k
    }

/*≡≡=---=≡≡≡≡≡≡≡=---=≡≡*\
||   RUNTIME SUPPORT   ||
\*≡==----=≡≡≡≡≡=----==≡*/

/** Parsers used by the generated parsers, resolving operands and results.
  *
  * Their types are refined unchecked to the ADT's field types, as is the case
  * for any structured operation: those are checked by its verification.
  */
object FormatParsing:

  def operand[A <: Attribute](name: String, typ: Attribute)(using
      P[Any],
      Parser,
  ): P[Operand[A]] =
    operandP(name, typ).asInstanceOf[P[Operand[A]]]

  def result[A <: Attribute](name: String, typ: Attribute)(using
      P[Any],
      Parser,
  ): P[Result[A]] =
    resultP(name, typ).asInstanceOf[P[Result[A]]]

  /** Resolve an optional construct from its optional name and type. */
  inline def optional[V](
      name: Option[String],
      typ: Option[Attribute],
      kind: String,
  )(inline one: (String, Attribute) => P[V])(using P[Any]): P[Option[V]] =
    (name, typ) match
      case (Some(name), Some(typ)) => one(name, typ).map(Some(_))
      case (None, None)            => Pass(None)
      case _                       => countFailure(kind, name.size, typ.size)

  /** Resolve a variadic construct from its names and types. */
  inline def variadic[V](
      names: Seq[String],
      types: Seq[Attribute],
      kind: String,
  )(inline one: (String, Attribute) => P[V])(using P[Any]): P[Seq[V]] =
    if names.length != types.length then
      countFailure(kind, names.length, types.length)
    else
      names.indices.foldLeft(Pass(Vector.empty[V]))((resolved, i) =>
        resolved
          .flatMapX(resolved => one(names(i), types(i)).map(resolved :+ _))
      )

  def countFailure(kind: String, names: Int, types: Int)(using P[Any]) =
    Fail(
      s"Number of ${kind}s ($names) does not match the number of the corresponding $kind types ($types)."
    )

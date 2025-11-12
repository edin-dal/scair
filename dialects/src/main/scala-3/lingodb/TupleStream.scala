package scair.dialects.LingoDB.TupleStream

import fastparse.*
import scair.AttrParser
import scair.AttrParser.whitespace
import scair.Parser
import scair.Parser.ValueId
import scair.Parser.orElse
import scair.Printer
import scair.clair.macros.DerivedOperation
import scair.clair.macros.summonDialect
import scair.dialects.builtin.*
import scair.ir.*

///////////
// TYPES //
///////////

// ==-----== //
//   Tuple   //
// ==-----== //

object TupleType extends AttributeCompanion:
  override def name: String = "tuples.tuple"

  override def parse[$: P](p: AttrParser) =
    P(("<" ~/ p.Type.rep(sep = ",") ~ ">").orElse(Seq()))
      .map(TupleType(_))

case class TupleType(val tupleVals: Seq[Attribute])
    extends ParametrizedAttribute
    with TypeAttribute:

  override def name: String = "tuples.tuple"
  override def parameters: Seq[Attribute | Seq[Attribute]] = tupleVals

  override def custom_verify(): Either[String, Unit] =
    if tupleVals.length != 0 && tupleVals.length != 2 then
      Left("TupleStream Tuple must contain 2 elements only.")
    else Right(())

// ==-----------== //
//   TupleStream   //
// ==-----------== //

object TupleStreamType extends AttributeCompanion:
  override def name: String = "tuples.tuplestream"

  override def parse[$: P](p: AttrParser) =
    P(("<" ~/ p.Type.rep(sep = ",") ~ ">").orElse(Seq()))
      .map(TupleStreamType(_))

case class TupleStreamType(val tuples: Seq[Attribute])
    extends ParametrizedAttribute
    with TypeAttribute:

  override def name: String = "tuples.tuplestream"
  override def parameters: Seq[Attribute | Seq[Attribute]] = tuples

  override def custom_verify(): Either[String, Unit] =

    lazy val verifyTuples: Int => Either[String, Unit] = (i: Int) =>
      if i == tuples.length then Right(())

      tuples(i) match
        case x: TupleType =>
          x.custom_verify() match
            case Right(_)  => verifyTuples(i + 1)
            case Left(err) => Left(err)
        case _ =>
          Left(
            "TupleStream must only contain TupleStream Tuple attributes."
          )

    verifyTuples(0)

////////////////
// ATTRIBUTES //
////////////////

// ==-------------== //
//   ColumnDefAttr   //
// ==-------------== //

object ColumnDefAttr extends AttributeCompanion:
  override def name: String = "tuples.column_def"

  override def parse[$: P](parser: AttrParser): P[Attribute] = P(
    parser.SymbolRefAttrP ~ "(" ~ "{" ~ parser.AttributeEntry ~ "}" ~ ")"
  ).map((x, y) => ColumnDefAttr(x.asInstanceOf[SymbolRefAttr], y._2))

case class ColumnDefAttr(val refName: SymbolRefAttr, val typ: Attribute)
    extends ParametrizedAttribute:

  override def name: String = "tuples.column_def"
  override def parameters: Seq[Attribute | Seq[Attribute]] = Seq(refName, typ)

  override def custom_print(p: Printer) =
    p.print(refName, "({type = ", typ, "})")(using indentLevel = 0)

// ==-------------== //
//   ColumnRefAttr   //
// ==-------------== //

object ColumnRefAttr extends AttributeCompanion:
  override def name: String = "tuples.column_ref"

  override def parse[$: P](parser: AttrParser): P[Attribute] =
    P(parser.SymbolRefAttrP).map(x =>
      ColumnRefAttr(x.asInstanceOf[SymbolRefAttr])
    )

case class ColumnRefAttr(val refName: SymbolRefAttr)
    extends ParametrizedAttribute:
  override def name: String = "tuples.column_ref"
  override def parameters: Seq[Attribute | Seq[Attribute]] = Seq(refName)
  override def custom_print(p: Printer) = p.print(refName)

////////////////
// OPERATIONS //
////////////////

// ==--------== //
//   ReturnOp   //
// ==--------== //

object ReturnOp:
  def name: String = "tuples.return"

  // ==--- Custom Parsing ---== //
  private def makeResults(
      x: Option[(Seq[String], Seq[Attribute])]
  ): (Seq[String], Seq[Attribute]) = x match
    case Some((y, z)) => (y, z)
    case None         => (Seq(), Seq())

  def parse[$: P](
      parser: Parser,
      resNames: Seq[String]
  ): P[Operation] = P(
    parser.OptionalAttributes ~ (ValueId.rep(sep = ",")
      ~ ":" ~
      parser.Type.rep(sep = ",")).orElse((Seq(), Seq()))
  ).map((x: Map[String, Attribute], y: (Seq[String], Seq[Attribute])) =>
    parser.generateOperation(
      opName = name,
      operandsNames = y._1,
      operandsTypes = y._2,
      resultsNames = resNames,
      attributes = x
    )
  )

  // ==----------------------== //

case class ReturnOp(
    _results: Seq[Operand[Attribute]]
) extends DerivedOperation["tuples.return", ReturnOp]
    with IsTerminator

// ==-----------== //
//   GetColumnOp   //
// ==-----------== //

object GetColumnOp:
  def name: String = "tuples.getcol"

  // ==--- Custom Parsing ---== //
  def parse[$: P](
      parser: Parser,
      resNames: Seq[String]
  ): P[Operation] = P(
    ValueId ~ ColumnRefAttr.parse(parser) ~ ":" ~
      parser.Type ~ parser.OptionalAttributes
  ).map(
    (
        x: String,
        y: Attribute,
        z: Attribute,
        w: Map[String, Attribute]
    ) =>
      val operand_type = parser.currentScope.valueMap(x).typ
      parser.generateOperation(
        opName = name,
        operandsNames = Seq(x),
        operandsTypes = Seq(operand_type),
        resultsNames = resNames,
        resultsTypes = Seq(z),
        attributes = w + ("attr" -> y)
      )
  )

  // ==----------------------== //

case class GetColumnOp(
    attr: ColumnRefAttr,
    tuple: TupleType,
    res: Result[Attribute]
) extends DerivedOperation["tuples.getcol", GetColumnOp]

/////////////
// DIALECT //
/////////////

val TupleStreamDialect: Dialect =
  summonDialect[EmptyTuple, (ReturnOp, GetColumnOp)](
    Seq(TupleType, TupleStreamType, ColumnDefAttr, ColumnRefAttr)
  )

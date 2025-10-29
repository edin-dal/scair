package scair.dialects.LingoDB.TupleStream

import fastparse.*
import scair.AttrParser
import scair.AttrParser.whitespace
import scair.Parser
import scair.Parser.ValueId
import scair.Parser.orElse
import scair.Printer
import scair.dialects.builtin.*
import scair.ir.*

///////////
// TYPES //
///////////

// ==-----== //
//   Tuple   //
// ==-----== //

object TupleStreamTuple extends AttributeCompanion:
  override def name: String = "tuples.tuple"

  override def parse[$: P](p: AttrParser) =
    P(("<" ~/ p.Type.rep(sep = ",") ~ ">").orElse(Seq()))
      .map(TupleStreamTuple(_))

case class TupleStreamTuple(val tupleVals: Seq[Attribute])
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

object TupleStream extends AttributeCompanion:
  override def name: String = "tuples.tuplestream"

  override def parse[$: P](p: AttrParser) =
    P(("<" ~/ p.Type.rep(sep = ",") ~ ">").orElse(Seq())).map(TupleStream(_))

case class TupleStream(val tuples: Seq[Attribute])
    extends ParametrizedAttribute
    with TypeAttribute:

  override def name: String = "tuples.tuplestream"
  override def parameters: Seq[Attribute | Seq[Attribute]] = tuples

  override def custom_verify(): Either[String, Unit] =

    lazy val verifyTuples: Int => Either[String, Unit] = (i: Int) =>
      if i == tuples.length then Right(())

      tuples(i) match
        case x: TupleStreamTuple =>
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

object ReturnOp extends OperationCompanion:
  override def name: String = "tuples.return"

  // ==--- Custom Parsing ---== //
  private def makeResults(
      x: Option[(Seq[String], Seq[Attribute])]
  ): (Seq[String], Seq[Attribute]) = x match
    case Some((y, z)) => (y, z)
    case None         => (Seq(), Seq())

  override def parse[$: P](
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
    override val operands: Seq[Value[Attribute]] = Seq(),
    override val successors: Seq[Block] = Seq(),
    override val results: Seq[Result[Attribute]] = Seq(),
    override val regions: Seq[Region] = Seq(),
    override val properties: Map[String, Attribute] = Map(),
    override val attributes: DictType[String, Attribute] = DictType()
) extends BaseOperation(
      name = "tuples.return",
      operands,
      successors,
      results,
      regions,
      properties,
      attributes
    )
    with IsTerminator:

  override def custom_verify(): Either[String, Operation] = (
    operands.length,
    successors.length,
    regions.length,
    properties.size
  ) match
    case (0, 0, 0, 0) => Right(this)
    case _            =>
      Left(
        "ReturnOp Operation must contain only results and an attribute dictionary."
      )

// ==-----------== //
//   GetColumnOp   //
// ==-----------== //

object GetColumnOp extends OperationCompanion:
  override def name: String = "tuples.getcol"

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
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
    override val operands: Seq[Value[Attribute]],
    override val successors: Seq[Block],
    override val results: Seq[Result[Attribute]],
    override val regions: Seq[Region],
    override val properties: Map[String, Attribute],
    override val attributes: DictType[String, Attribute]
) extends BaseOperation(
      name = "tuples.getcol",
      operands,
      successors,
      results,
      regions,
      properties,
      attributes
    ):

  override def custom_verify(): Either[String, Operation] = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    properties.size
  ) match
    case (1, 0, 1, 0, 0) =>
      {
        operands(0).typ match
          case x: TupleStreamTuple =>
            x.custom_verify() match
              case Right(value) => Right(this)
              case Left(err)    => Left(err)
          case _ =>
            Left(
              "GetColumnOp Operation must contain an operand of type TupleStreamTuple."
            )
      }.flatMap(_ =>
        attributes.get("attr") match
          case Some(x) =>
            x match
              case _: ColumnRefAttr => Right(this)
              case _                =>
                Left(
                  "GetColumnOp Operation must contain a ColumnRefAttr Attribute."
                )
          case None =>
            Left(
              "GetColumnOp Operation must contain a ColumnRefAttr Attribute."
            )
      )
    case _ =>
      Left(
        "GetColumnOp Operation must contain only 2 operands, 1 result and an attribute dictionary."
      )

/////////////
// DIALECT //
/////////////

val TupleStreamDialect: Dialect =
  new Dialect(
    operations = Seq(ReturnOp, GetColumnOp),
    attributes =
      Seq(TupleStreamTuple, TupleStream, ColumnDefAttr, ColumnRefAttr)
  )

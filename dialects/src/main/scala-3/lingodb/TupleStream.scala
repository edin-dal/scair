package scair.dialects.LingoDB.TupleStream

import fastparse.*
import scair.AttrParser
import scair.Parser
import scair.Parser.ValueId
import scair.Parser.orElse
import scair.Parser.whitespace
import scair.dialects.builtin.*
import scair.ir.*

import scala.collection.immutable
import scala.collection.mutable

///////////
// TYPES //
///////////

// ==-----== //
//   Tuple   //
// ==-----== //

object TupleStreamTuple extends AttributeObject {
  override def name: String = "tuples.tuple"

  override def parse[$: P](p: AttrParser) =
    P(("<" ~/ p.Type.rep(sep = ",") ~ ">").orElse(Seq()))
      .map(TupleStreamTuple(_))

}

case class TupleStreamTuple(val tupleVals: Seq[Attribute])
    extends ParametrizedAttribute(
      name = "tuples.tuple",
      parameters = tupleVals
    )
    with TypeAttribute {

  override def custom_verify(): Either[Unit, String] = {
    if (tupleVals.length != 0 && tupleVals.length != 2) {
      Right("TupleStream Tuple must contain 2 elements only.")
    } else {
      Left(())
    }
  }

}

// ==-----------== //
//   TupleStream   //
// ==-----------== //

object TupleStream extends AttributeObject {
  override def name: String = "tuples.tuplestream"

  override def parse[$: P](p: AttrParser) =
    P(("<" ~/ p.Type.rep(sep = ",") ~ ">").orElse(Seq())).map(TupleStream(_))

}

case class TupleStream(val tuples: Seq[Attribute])
    extends ParametrizedAttribute(
      name = "tuples.tuplestream",
      parameters = tuples
    )
    with TypeAttribute {

  override def custom_verify(): Either[Unit, String] = {

    lazy val verifyTuples: Int => Either[Unit, String] = { (i: Int) =>
      if (i == tuples.length) then Left(())

      tuples(i) match {
        case x: TupleStreamTuple =>
          x.custom_verify() match {
            case Left(_)    => verifyTuples(i + 1)
            case Right(err) => Right(err)
          }
        case _ =>
          Right(
            "TupleStream must only contain TupleStream Tuple attributes."
          )
      }
    }

    verifyTuples(0)
  }

}

////////////////
// ATTRIBUTES //
////////////////

// ==-------------== //
//   ColumnDefAttr   //
// ==-------------== //

object ColumnDefAttr extends AttributeObject {
  override def name: String = "tuples.column_def"

  override def parse[$: P](parser: AttrParser): P[Attribute] = P(
    parser.SymbolRefAttrP ~ "(" ~ "{" ~ parser.AttributeEntry ~ "}" ~ ")"
  ).map((x, y) => ColumnDefAttr(x.asInstanceOf[SymbolRefAttr], y._2))

}

case class ColumnDefAttr(val refName: SymbolRefAttr, val typ: Attribute)
    extends ParametrizedAttribute(
      name = "tuples.column_def"
    ) {

  override def custom_print =
    s"${refName.custom_print}({type = ${typ.custom_print}})"

}

// ==-------------== //
//   ColumnRefAttr   //
// ==-------------== //

object ColumnRefAttr extends AttributeObject {
  override def name: String = "tuples.column_ref"

  override def parse[$: P](parser: AttrParser): P[Attribute] =
    P(parser.SymbolRefAttrP).map(x =>
      ColumnRefAttr(x.asInstanceOf[SymbolRefAttr])
    )

}

case class ColumnRefAttr(val refName: SymbolRefAttr)
    extends ParametrizedAttribute(
      name = "tuples.column_ref"
    ) {
  override def custom_print = s"${refName.custom_print}"
}

////////////////
// OPERATIONS //
////////////////

// ==--------== //
//   ReturnOp   //
// ==--------== //

object ReturnOp extends OperationCompanion {
  override def name: String = "tuples.return"

  // ==--- Custom Parsing ---== //
  private def makeResults(
      x: Option[(Seq[String], Seq[Attribute])]
  ): (Seq[String], Seq[Attribute]) = x match {
    case Some((y, z)) => (y, z)
    case None         => (Seq(), Seq())
  }

  override def parse[$: P](
      parser: Parser
  ): P[Operation] = P(
    parser.OptionalAttributes ~ (ValueId.rep(sep = ",")
      ~ ":" ~
      parser.Type.rep(sep = ",")).orElse((Seq(), Seq()))
  ).map((x: Map[String, Attribute], y: (Seq[String], Seq[Attribute])) =>
    parser.generateOperation(
      opName = name,
      operandsNames = y._1,
      operandsTypes = y._2,
      attributes = x
    )
  )
  // ==----------------------== //

}

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
    ) {

  override def custom_verify(): Either[Operation, String] = (
    operands.length,
    successors.length,
    regions.length,
    properties.size
  ) match {
    case (0, 0, 0, 0) => Left(this)
    case _ =>
      Right(
        "ReturnOp Operation must contain only results and an attribute dictionary."
      )
  }

}

// ==-----------== //
//   GetColumnOp   //
// ==-----------== //

object GetColumnOp extends OperationCompanion {
  override def name: String = "tuples.getcol"

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      parser: Parser
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
        resultsTypes = Seq(z),
        attributes = w + ("attr" -> y)
      )
  )
  // ==----------------------== //

}

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
    ) {

  override def custom_verify(): Either[Operation, String] = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    properties.size
  ) match {
    case (1, 0, 1, 0, 0) =>
      {
        operands(0).typ match {
          case x: TupleStreamTuple =>
            x.custom_verify() match {
              case Left(value) => Left(this)
              case Right(err)  => Right(err)
            }
          case _ =>
            Right(
              "GetColumnOp Operation must contain an operand of type TupleStreamTuple."
            )
        }
      }.orElse({
        attributes.get("attr") match {
          case Some(x) =>
            x match {
              case _: ColumnRefAttr => Left(this)
              case _ =>
                Right(
                  "GetColumnOp Operation must contain a ColumnRefAttr Attribute."
                )
            }
          case None =>
            Right(
              "GetColumnOp Operation must contain a ColumnRefAttr Attribute."
            )
        }
      })
    case _ =>
      Right(
        "GetColumnOp Operation must contain only 2 operands, 1 result and an attribute dictionary."
      )
  }

}

/////////////
// DIALECT //
/////////////

val TupleStreamDialect: Dialect =
  new Dialect(
    operations = Seq(ReturnOp, GetColumnOp),
    attributes =
      Seq(TupleStreamTuple, TupleStream, ColumnDefAttr, ColumnRefAttr)
  )

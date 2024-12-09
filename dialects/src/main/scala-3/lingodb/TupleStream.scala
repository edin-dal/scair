package scair.dialects.LingoDB.TupleStream

import fastparse.*
import scair.AttrParser
import scair.Parser
import scair.Parser.ValueId
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
  override def factory = TupleStreamTuple.apply
}

case class TupleStreamTuple(val tupleVals: Seq[Attribute])
    extends ParametrizedAttribute(
      name = "tuples.tuple",
      parameters = tupleVals
    )
    with TypeAttribute {

  override def custom_verify(): Unit = {
    if (tupleVals.length != 0 && tupleVals.length != 2) {
      throw new Exception("TupleStream Tuple must contain 2 elements only.")
    }
  }
}

// ==-----------== //
//   TupleStream   //
// ==-----------== //

object TupleStream extends AttributeObject {
  override def name: String = "tuples.tuplestream"
  override def factory = TupleStream.apply
}

case class TupleStream(val tuples: Seq[Attribute])
    extends ParametrizedAttribute(
      name = "tuples.tuplestream",
      parameters = tuples
    )
    with TypeAttribute {

  override def custom_verify(): Unit = {
    for (param <- tuples) {
      param match {
        case _: TupleStreamTuple =>
        case _ =>
          throw new Exception(
            "TupleStream must only contain TupleStream Tuple attributes."
          )
      }
    }
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

object ReturnOp extends OperationObject {
  override def name: String = "tuples.return"
  override def factory = ReturnOp.apply

  // ==--- Custom Parsing ---== //
  private def makeResults(
      x: Option[(Seq[String], Seq[Attribute])]
  ): (Seq[String], Seq[Attribute]) = x match {
    case Some((y, z)) => (y, z)
    case None         => (Seq(), Seq())
  }
  override def parse[$: P](
      resNames: Seq[String],
      parser: Parser
  ): P[Operation] = P(
    parser.DictionaryAttribute.?.map(Parser.optionlessSeq) ~ (ValueId.rep(sep =
      ","
    )
      ~ ":" ~
      parser.Type.rep(sep = ",")).?.map(makeResults)
  ).map((x: Seq[(String, Attribute)], y: (Seq[String], Seq[Attribute])) =>
    parser.verifyCustomOp(
      opGen = constructOp,
      opName = name,
      operandNames = y._1,
      operandTypes = y._2,
      dictAttrs = x
    )
  )
  // ==----------------------== //
}

case class ReturnOp(
    override val operands: ListType[Value[Attribute]],
    override val successors: ListType[Block],
    override val results: ListType[Value[Attribute]],
    override val regions: ListType[Region],
    override val dictionaryProperties: DictType[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute]
) extends RegisteredOperation(name = "tuples.return") {

  override def custom_verify(): Unit = (
    operands.length,
    successors.length,
    regions.length,
    dictionaryProperties.size
  ) match {
    case (0, 0, 0, 0) =>
    case _ =>
      throw new Exception(
        "ReturnOp Operation must contain only results and an attribute dictionary."
      )
  }
}

// ==-----------== //
//   GetColumnOp   //
// ==-----------== //

object GetColumnOp extends OperationObject {
  override def name: String = "tuples.getcol"
  override def factory = GetColumnOp.apply

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      resNames: Seq[String],
      parser: Parser
  ): P[Operation] = P(
    ValueId.rep(exactly = 1) ~ ColumnRefAttr.parse(parser) ~ ":" ~
      parser.Type ~ parser.DictionaryAttribute.?.map(Parser.optionlessSeq)
  ).map(
    (
        x: Seq[String],
        y: Attribute,
        z: Attribute,
        w: Seq[(String, Attribute)]
    ) =>
      parser.verifyCustomOp(
        opGen = factory,
        opName = name,
        operandNames = x,
        resultNames = resNames,
        resultTypes = Seq(z),
        dictAttrs = w :+ ("attr", y),
        noForwardOperandRef = 1
      )
  )
  // ==----------------------== //
}

case class GetColumnOp(
    override val operands: ListType[Value[Attribute]],
    override val successors: ListType[Block],
    override val results: ListType[Value[Attribute]],
    override val regions: ListType[Region],
    override val dictionaryProperties: DictType[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute]
) extends RegisteredOperation(name = "tuples.getcol") {

  override def custom_verify(): Unit = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    dictionaryProperties.size
  ) match {
    case (1, 0, 1, 0, 0) =>
      operands(0).typ match {
        case x: TupleStreamTuple => x.custom_verify()
        case _ =>
          throw new Exception(
            "GetColumnOp Operation must contain an operand of type TupleStreamTuple."
          )
      }
      dictionaryAttributes.get("attr") match {
        case Some(x) =>
          x match {
            case _: ColumnRefAttr =>
            case _ =>
              throw new Exception(
                "GetColumnOp Operation must contain a ColumnRefAttr Attribute."
              )
          }
        case None =>
          throw new Exception(
            "GetColumnOp Operation must contain a ColumnRefAttr Attribute."
          )
      }
    case _ =>
      throw new Exception(
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

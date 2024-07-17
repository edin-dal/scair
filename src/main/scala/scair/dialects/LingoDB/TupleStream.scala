package scair.dialects.LingoDB.TupleStream

import fastparse._
import scair.dialects.builtin._
import scala.collection.immutable
import scair.dialects.irdl.{Operand, OpResult}
import scair.Parser.{whitespace, ValueId, Type, DictionaryAttribute}
import scair.{
  RegisteredOperation,
  Region,
  Block,
  Value,
  Attribute,
  TypeAttribute,
  ParametrizedAttribute,
  DialectAttribute,
  DialectOperation,
  Dialect,
  Parser,
  Operation,
  AttrParser
}

///////////
// TYPES //
///////////

// ==-----== //
//   Tuple   //
// ==-----== //

object TupleStreamTuple extends DialectAttribute {
  override def name: String = "tuples.tuple"
  override def parse[$: P]: P[Attribute] =
    P("<" ~ Type.rep(sep = ",") ~ ">").map(TupleStreamTuple(_))
}

case class TupleStreamTuple(val tupleVals: Seq[Attribute])
    extends ParametrizedAttribute(
      name = "tuples.tuple",
      parameters = tupleVals
    )
    with TypeAttribute {

  override def verify(): Unit = {
    if (tupleVals.length != 2) {
      throw new Exception("TupleStream Tuple must contain 2 elements only.")
    }
  }
  override def toString =
    s"${prefix}${name}<${tupleVals.map(x => x.toString).mkString(", ")}>"
}

// ==-----------== //
//   TupleStream   //
// ==-----------== //

object TupleStream extends DialectAttribute {
  override def name: String = "tuples.tuplestream"
  override def parse[$: P]: P[Attribute] =
    P("<" ~ Type.rep(sep = ",") ~ ">").map(TupleStream(_))
}

case class TupleStream(val tuples: Seq[Attribute])
    extends ParametrizedAttribute(
      name = "tuples.tuplestream",
      parameters = tuples
    )
    with TypeAttribute {

  override def verify(): Unit = {
    for (param <- tuples) {
      if (!param.isInstanceOf[TupleStreamTuple]) {
        throw new Exception(
          "TupleStream must only contain TupleStream Tuple attributes."
        )
      }
    }
  }
  override def toString =
    s"${prefix}${name}<${tuples.map(x => x.toString).mkString(", ")}>"
}

////////////////
// ATTRIBUTES //
////////////////

// ==-------------== //
//   ColumnDefAttr   //
// ==-------------== //

object ColumnDefAttr extends DialectAttribute {
  override def name: String = "tuples.column_def"
  override def parse[$: P]: P[Attribute] =
    P("<" ~ Type ~ "," ~ Type ~ ">").map(ColumnDefAttr(_, _))
}

case class ColumnDefAttr(val refName: Attribute, val fromExisting: Attribute)
    extends ParametrizedAttribute(
      name = "tuples.column_def",
      Seq(refName, fromExisting)
    ) {

  override def verify(): Unit = {
    if (!refName.isInstanceOf[SymbolRefAttr]) {
      throw new Exception(
        "ColumnDefAttr's name must be of SymbolRefAttr Attribute."
      )
    }
  }
  override def toString =
    s"${prefix}${name}<${refName}, ${fromExisting}>"
}

// ==-------------== //
//   ColumnRefAttr   //
// ==-------------== //

object ColumnRefAttr extends DialectAttribute {
  override def name: String = "tuples.column_ref"
  override def parse[$: P]: P[Attribute] =
    P("<" ~ Type ~ ">").map(ColumnRefAttr(_))
}

case class ColumnRefAttr(val refName: Attribute)
    extends ParametrizedAttribute(
      name = "tuples.column_ref",
      Seq(refName)
    ) {

  override def verify(): Unit = refName match {
    case _: SymbolRefAttr =>
    case _ =>
      throw new Exception(
        "ColumnRefAttr's name must be of SymbolRefAttr Attribute."
      )
  }
  override def toString =
    s"${prefix}${name}<${refName}>"
}

////////////////
// OPERATIONS //
////////////////

// ==--------== //
//   ReturnOp   //
// ==--------== //

object ReturnOp extends DialectOperation {
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
    DictionaryAttribute.?.map(Parser.optionlessSeq) ~ (ValueId.rep(sep = ",")
      ~ ":" ~
      Type.rep(sep = ",")).?.map(makeResults)
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
    override val operands: collection.mutable.ArrayBuffer[Value[Attribute]],
    override val successors: collection.mutable.ArrayBuffer[Block],
    override val results: Seq[Value[Attribute]],
    override val regions: Seq[Region],
    override val dictionaryProperties: immutable.Map[String, Attribute],
    override val dictionaryAttributes: immutable.Map[String, Attribute]
) extends RegisteredOperation(name = "tuples.return") {

  override def verify(): Unit = (
    operands.length,
    successors.length,
    regions.length,
    dictionaryProperties.size
  ) match {
    case (0, 0, 0, 0) =>
      for (x <- results) yield x.typ.verify()
      for ((x, y) <- dictionaryAttributes) yield y.verify()
    case _ =>
      throw new Exception(
        "ReturnOp Operation must contain only results and an attribute dictionary."
      )
  }
}

// ==-----------== //
//   GetColumnOp   //
// ==-----------== //

object GetColumnOp extends DialectOperation {
  override def name: String = "tuples.getcol"
  override def factory = GetColumnOp.apply

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      resNames: Seq[String],
      parser: Parser
  ): P[Operation] = P(
    ValueId.rep(exactly = 1) ~ AttrParser.SymbolRefAttrP ~ ":" ~
      Type ~ DictionaryAttribute.?.map(Parser.optionlessSeq)
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
    override val operands: collection.mutable.ArrayBuffer[Value[Attribute]],
    override val successors: collection.mutable.ArrayBuffer[Block],
    override val results: Seq[Value[Attribute]],
    override val regions: Seq[Region],
    override val dictionaryProperties: immutable.Map[String, Attribute],
    override val dictionaryAttributes: immutable.Map[String, Attribute]
) extends RegisteredOperation(name = "tuples.getcol") {

  override def verify(): Unit = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    dictionaryProperties.size
  ) match {
    case (2, 0, 1, 0, 0) =>
      operands(0).typ.verify()
      operands(1).typ.verify()
      results(0).typ.verify()
      for ((x, y) <- dictionaryAttributes) yield y.verify()
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

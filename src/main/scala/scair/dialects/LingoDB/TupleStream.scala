package scair.dialects.LingoDB.TupleStream

import fastparse._
import scair.dialects.builtin._
import scala.collection.immutable
import scair.dialects.irdl.{Operand, OpResult}
import scair.Parser.{whitespace, Type}
import scair.{
  RegisteredOperation,
  Region,
  Block,
  Value,
  Attribute,
  TypeAttribute,
  ParametrizedAttribute,
  DataAttribute,
  DialectAttribute,
  DialectOperation,
  Dialect,
  Printer,
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
    P("<" ~ Type.rep(sep = ",") ~ ">").map(TupleStreamTuple(_*))
}

case class TupleStreamTuple(val tupleVals: Attribute*)
    extends ParametrizedAttribute(
      name = "tuples.tuple",
      parameters = tupleVals: _*
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
    P("<" ~ Type.rep(sep = ",") ~ ">").map(TupleStream(_*))
}

case class TupleStream(val tuples: Attribute*)
    extends ParametrizedAttribute(
      name = "tuples.tuplestream",
      parameters = tuples: _*
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
      refName,
      fromExisting
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
      refName
    ) {

  override def verify(): Unit = {
    if (!refName.isInstanceOf[SymbolRefAttr]) {
      throw new Exception(
        "ColumnRefAttr's name must be of SymbolRefAttr Attribute."
      )
    }
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
  override def constructOp(
      operands: collection.mutable.ArrayBuffer[Value[Attribute]] =
        collection.mutable.ArrayBuffer(),
      successors: collection.mutable.ArrayBuffer[Block] =
        collection.mutable.ArrayBuffer(),
      results: Seq[Value[Attribute]] = Seq[Value[Attribute]](),
      regions: Seq[Region] = Seq[Region](),
      dictionaryProperties: immutable.Map[String, Attribute] =
        immutable.Map.empty[String, Attribute],
      dictionaryAttributes: immutable.Map[String, Attribute] =
        immutable.Map.empty[String, Attribute]
  ): ReturnOp = ReturnOp(
    operands,
    successors,
    results,
    regions,
    dictionaryProperties,
    dictionaryAttributes
  )
}

case class ReturnOp(
    override val operands: collection.mutable.ArrayBuffer[Value[Attribute]] =
      collection.mutable.ArrayBuffer(),
    override val successors: collection.mutable.ArrayBuffer[Block] =
      collection.mutable.ArrayBuffer(),
    override val results: Seq[Value[Attribute]] = Seq[Value[Attribute]](),
    override val regions: Seq[Region] = Seq[Region](),
    override val dictionaryProperties: immutable.Map[String, Attribute] =
      immutable.Map.empty[String, Attribute],
    override val dictionaryAttributes: immutable.Map[String, Attribute] =
      immutable.Map.empty[String, Attribute]
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

// ==--------== //
//   ReturnOp   //
// ==--------== //

object GetColumnOp extends DialectOperation {
  override def name: String = "tuples.getcol"
  override def constructOp(
      operands: collection.mutable.ArrayBuffer[Value[Attribute]] =
        collection.mutable.ArrayBuffer(),
      successors: collection.mutable.ArrayBuffer[Block] =
        collection.mutable.ArrayBuffer(),
      results: Seq[Value[Attribute]] = Seq[Value[Attribute]](),
      regions: Seq[Region] = Seq[Region](),
      dictionaryProperties: immutable.Map[String, Attribute] =
        immutable.Map.empty[String, Attribute],
      dictionaryAttributes: immutable.Map[String, Attribute] =
        immutable.Map.empty[String, Attribute]
  ): GetColumnOp = GetColumnOp(
    operands,
    successors,
    results,
    regions,
    dictionaryProperties,
    dictionaryAttributes
  )
}

case class GetColumnOp(
    override val operands: collection.mutable.ArrayBuffer[Value[Attribute]] =
      collection.mutable.ArrayBuffer(),
    override val successors: collection.mutable.ArrayBuffer[Block] =
      collection.mutable.ArrayBuffer(),
    override val results: Seq[Value[Attribute]] = Seq[Value[Attribute]](),
    override val regions: Seq[Region] = Seq[Region](),
    override val dictionaryProperties: immutable.Map[String, Attribute] =
      immutable.Map.empty[String, Attribute],
    override val dictionaryAttributes: immutable.Map[String, Attribute] =
      immutable.Map.empty[String, Attribute]
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
        "ReturnOp Operation must contain only results and an attribute dictionary."
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

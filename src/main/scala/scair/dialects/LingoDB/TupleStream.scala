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

case class ColumnDefAttr()
    extends ParametrizedAttribute(
      name = "tuples.columndef"
    )

case class ColumnRefAttr()
    extends ParametrizedAttribute(
      name = "tuples.columnref"
    )

////////////////
// OPERATIONS //
////////////////

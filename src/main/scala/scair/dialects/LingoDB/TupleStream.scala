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

// ///////////
// // TYPES //
// ///////////

// case class TupleStream extends ParametrizedAttribute(
//   name = "tuples.tuplestream"
//   parameters = Seq()
// ) with TypeAttribute

// case class TupleStreamTuple extends ParametrizedAttribute(
//   name = "tuples.tuple"
//   parameters = Seq()
// ) with TypeAttribute

// ////////////////
// // ATTRIBUTES //
// ////////////////

// case class ColumnDefAttr() extends ParametrizedAttribute(
//   name = "tuples.columndef"
//   parameters = Seq()
// ) with TypeAttribute

// case class TupleStreamTuple extends ParametrizedAttribute(
//   name = "tuples.tuple"
//   parameters = Seq()
// ) with TypeAttribute

// ////////////////
// // OPERATIONS //
// ////////////////

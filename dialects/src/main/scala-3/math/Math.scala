package scair.dialects.math

import fastparse.*
import scair.Parser
import scair.ir.*

import scala.NotImplementedError
import scala.collection.immutable
import scala.collection.mutable

////////////////
// OPERATIONS //
////////////////

// ==--------== //
//   AbsfOp   //
// ==--------== //

object AbsfOp extends OperationObject {
  override def name: String = "math.absf"
  override def factory = AbsfOp.apply

  override def parse[$: P](
      resNames: Seq[String],
      parser: Parser
  ): P[Operation] =
    throw new NotImplementedError("This custom parse needs to be implemented!")
  // ==----------------------== //

}

case class AbsfOp(
    override val operands: ListType[Value[Attribute]],
    override val successors: ListType[Block],
    override val results: ListType[Value[Attribute]],
    override val regions: ListType[Region],
    override val dictionaryProperties: DictType[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute]
) extends RegisteredOperation(name = "math.absf") {

  override def custom_verify(): Unit = (
    operands.length,
    results.length,
    successors.length,
    regions.length,
    dictionaryProperties.size
  ) match {
    case (1, 1, 0, 0, 0) =>
    case _ =>
      throw new Exception(
        "AbsfOp must have 1 result and 1 operands."
      )
  }

}

// ==--------== //
//   FPowIOp   //
// ==--------== //

object FPowIOp extends OperationObject {
  override def name: String = "math.fpowi"
  override def factory = FPowIOp.apply

  override def parse[$: P](
      resNames: Seq[String],
      parser: Parser
  ): P[Operation] =
    throw new NotImplementedError("This custom parse needs to be implemented!")
  // ==----------------------== //

}

case class FPowIOp(
    override val operands: ListType[Value[Attribute]],
    override val successors: ListType[Block],
    override val results: ListType[Value[Attribute]],
    override val regions: ListType[Region],
    override val dictionaryProperties: DictType[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute]
) extends RegisteredOperation(name = "math.fpowi") {

  override def custom_verify(): Unit = (
    operands.length,
    results.length,
    successors.length,
    regions.length,
    dictionaryProperties.size
  ) match {
    case (2, 1, 0, 0, 0) =>
    case _ =>
      throw new Exception(
        "FPowIOp must have 1 result and 2 operands."
      )
  }

}
/////////////
// DIALECT //
/////////////

val MathDialect: Dialect =
  new Dialect(
    operations = Seq(AbsfOp, FPowIOp),
    attributes = Seq()
  )

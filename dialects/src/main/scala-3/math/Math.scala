package scair.dialects.math

import fastparse.*
import scair.Parser
import scair.Parser.whitespace
import scair.ir.*

import scala.collection.mutable

////////////////
// OPERATIONS //
////////////////

// ==--------== //
//   AbsfOp   //
// ==--------== //

object AbsfOp extends OperationCompanion {
  override def name: String = "math.absf"

  override def parse[$: P](
      parser: Parser
  ): P[Operation] = {
    P(
      "" ~ Parser.ValueUse ~ ":" ~ parser.Type
    ).map { case (operandName, type_) =>
      parser.generateOperation(
        opName = name,
        operandsNames = Seq(operandName),
        operandsTypes = Seq(type_),
        resultsTypes = Seq(type_)
      )
    }
  }

}

case class AbsfOp(
    override val operands: Seq[Value[Attribute]],
    override val successors: Seq[Block],
    override val results_types: Seq[Attribute],
    override val regions: Seq[Region],
    override val properties: Map[String, Attribute],
    override val attributes: DictType[String, Attribute]
) extends BaseOperation(
      name = "math.absf",
      operands,
      successors,
      results_types,
      regions,
      properties,
      attributes
    ) {

  override def custom_verify(): Either[Operation, String] = (
    operands.length,
    results.length,
    successors.length,
    regions.length,
    properties.size
  ) match {
    case (1, 1, 0, 0, 0) => Left(this)
    case _ =>
      Right(
        "AbsfOp must have 1 result and 1 operand."
      )
  }

}

// ==--------== //
//   FPowIOp   //
// ==--------== //

object FPowIOp extends OperationCompanion {
  override def name: String = "math.fpowi"

  override def parse[$: P](
      parser: Parser
  ): P[Operation] = {
    P(
      Parser.ValueUse ~ "," ~ Parser.ValueUse ~ ":" ~ parser.Type ~ "," ~ parser.Type
    ).map {
      case (
            operand1Name,
            operand2Name,
            operand1Type,
            operand2Type
          ) =>
        parser.generateOperation(
          opName = name,
          operandsNames = Seq(operand1Name, operand2Name),
          operandsTypes = Seq(operand1Type, operand2Type),
          resultsTypes = Seq(operand1Type)
        )
    }
  }

}

case class FPowIOp(
    override val operands: Seq[Value[Attribute]],
    override val successors: Seq[Block],
    override val results_types: Seq[Attribute],
    override val regions: Seq[Region],
    override val properties: Map[String, Attribute],
    override val attributes: DictType[String, Attribute]
) extends BaseOperation(
      name = "math.fpowi",
      operands,
      successors,
      results_types,
      regions,
      properties,
      attributes
    ) {

  override def custom_verify(): Either[Operation, String] = (
    operands.length,
    results.length,
    successors.length,
    regions.length,
    properties.size
  ) match {
    case (2, 1, 0, 0, 0) => Left(this)
    case _ =>
      Right(
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

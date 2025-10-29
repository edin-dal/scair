package scair.dialects.math

import fastparse.*
import scair.AttrParser.whitespace
import scair.Parser
import scair.ir.*

////////////////
// OPERATIONS //
////////////////

// ==--------== //
//   AbsfOp   //
// ==--------== //

object AbsfOp extends OperationCompanion:
  override def name: String = "math.absf"

  override def parse[$: P](
      parser: Parser,
      resNames: Seq[String]
  ): P[Operation] =
    P(
      "" ~ Parser.ValueUse ~ ":" ~ parser.Type
    ).map { case (operandName, type_) =>
      parser.generateOperation(
        opName = name,
        operandsNames = Seq(operandName),
        operandsTypes = Seq(type_),
        resultsNames = resNames,
        resultsTypes = Seq(type_)
      )
    }

case class AbsfOp(
    override val operands: Seq[Value[Attribute]],
    override val successors: Seq[Block],
    override val results: Seq[Result[Attribute]],
    override val regions: Seq[Region],
    override val properties: Map[String, Attribute],
    override val attributes: DictType[String, Attribute]
) extends BaseOperation(
      name = "math.absf",
      operands,
      successors,
      results,
      regions,
      properties,
      attributes
    )
    with NoMemoryEffect:

  override def custom_verify(): Either[String, Operation] = (
    operands.length,
    results.length,
    successors.length,
    regions.length,
    properties.size
  ) match
    case (1, 1, 0, 0, 0) => Right(this)
    case _               =>
      Left(
        "AbsfOp must have 1 result and 1 operand."
      )

// ==--------== //
//   FPowIOp   //
// ==--------== //

object FPowIOp extends OperationCompanion:
  override def name: String = "math.fpowi"

  override def parse[$: P](
      parser: Parser,
      resNames: Seq[String]
  ): P[Operation] =
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
          resultsNames = resNames,
          resultsTypes = Seq(operand1Type)
        )
    }

case class FPowIOp(
    override val operands: Seq[Value[Attribute]],
    override val successors: Seq[Block],
    override val results: Seq[Result[Attribute]],
    override val regions: Seq[Region],
    override val properties: Map[String, Attribute],
    override val attributes: DictType[String, Attribute]
) extends BaseOperation(
      name = "math.fpowi",
      operands,
      successors,
      results,
      regions,
      properties,
      attributes
    )
    with NoMemoryEffect:

  override def custom_verify(): Either[String, Operation] = (
    operands.length,
    results.length,
    successors.length,
    regions.length,
    properties.size
  ) match
    case (2, 1, 0, 0, 0) => Right(this)
    case _               =>
      Left(
        "FPowIOp must have 1 result and 2 operands."
      )

/////////////
// DIALECT //
/////////////

val MathDialect: Dialect =
  new Dialect(
    operations = Seq(AbsfOp, FPowIOp),
    attributes = Seq()
  )

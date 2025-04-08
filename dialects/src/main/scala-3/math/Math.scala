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

object AbsfOp extends MLIROperationObject {
  override def name: String = "math.absf"

  override def parse[$: P](
      parser: Parser
  ): P[MLIROperation] = {
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
    override val operands: ListType[Value[Attribute]],
    override val successors: ListType[Block],
    results_types: ListType[Attribute],
    override val regions: ListType[Region],
    override val dictionaryProperties: DictType[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute]
) extends MLIROperation(
      name = "math.absf",
      operands,
      successors,
      results_types,
      regions,
      dictionaryProperties,
      dictionaryAttributes
    ) {

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
        "AbsfOp must have 1 result and 1 operand."
      )
  }

}

// ==--------== //
//   FPowIOp   //
// ==--------== //

object FPowIOp extends MLIROperationObject {
  override def name: String = "math.fpowi"

  override def parse[$: P](
      parser: Parser
  ): P[MLIROperation] = {
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
    override val operands: ListType[Value[Attribute]],
    override val successors: ListType[Block],
    results_types: ListType[Attribute],
    override val regions: ListType[Region],
    override val dictionaryProperties: DictType[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute]
) extends MLIROperation(
      name = "math.fpowi",
      operands,
      successors,
      results_types,
      regions,
      dictionaryProperties,
      dictionaryAttributes
    ) {

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

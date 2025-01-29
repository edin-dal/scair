package scair.dialects.math

import fastparse.*
import scair.Parser
import scair.Parser.whitespace
import scair.ir.*

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
  override def factory: FactoryType = AbsfOp.apply

  override def parse[$: P](
      resNames: Seq[String],
      parser: Parser
  ): P[Operation] = {
    P(
      "" ~ Parser.ValueUse ~ ":" ~ parser.Type
    ).map { case (operandName, type_) =>
      if (resNames.length != 1) {
        throw new Exception(
          s"AbsfOp must produce exactly one result, but found ${resNames.length}."
        )
      }

      parser.verifyCustomOp(
        opGen = AbsfOp.apply,
        opName = name,
        operandNames = Seq(operandName),
        operandTypes = Seq(type_),
        resultNames = resNames,
        resultTypes = Seq(type_)
      )
    }
  }

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
        "AbsfOp must have 1 result and 1 operand."
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
        if (resNames.length != 1) {
          throw new Exception(
            s"FPowIOp must produce exactly one result, but found ${resNames.length}."
          )
        }

        parser.verifyCustomOp(
          opGen = FPowIOp.apply,
          opName = name,
          operandNames = Seq(operand1Name, operand2Name),
          operandTypes = Seq(operand1Type, operand2Type),
          resultNames = resNames,
          resultTypes = Seq(operand1Type)
        )
    }
  }

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

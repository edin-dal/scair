package scair.dialects.CMath.cmath

import fastparse._
import scair.dialects.builtin._
import scala.collection.immutable
import scala.collection.mutable
import scair.Parser.{whitespace, ValueId, Type}
import scair.Parser.{whitespace, ValueId, Type, E}
import scair.AttrParser.{Float32TypeP, Float64TypeP}
import scair.{Printer, Parser}
import scair.ir._

import scair.exceptions.VerifyException

//////////////////
// COMPLEX TYPE //
//////////////////

object ComplexType extends DialectAttribute {
  override def name: String = "cmath.complex"
  override def factory = ComplexType.apply
}

case class ComplexType(val body: Seq[Attribute])
    extends ParametrizedAttribute("cmath.complex", body)
    with TypeAttribute {

  override def custom_verify(): Unit =
    if (body.length != 1) {
      throw new VerifyException(
        "TupleStream Tuple must contain 1 elements only."
      )
    } else
      body(0) match {
        case Float32Type =>
        case Float64Type =>
        case _ =>
          throw new VerifyException(
            "Complex type must be constructed with either 'f32' or 'f64' attribute."
          )
      }
}

////////////////////
// NORM OPERATION //
////////////////////

object Norm extends DialectOperation {
  override def name: String = "cmath.norm"
  override def factory: FactoryType = Norm.apply

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      resNames: Seq[String],
      parser: Parser
  ): P[Operation] = P(
    "(" ~ ValueId ~ ":" ~ ComplexType.parse ~ ")" ~ "=>" ~ (Float32TypeP | Float64TypeP)
  ).map(
    (
        operand: String,
        itstype: Attribute,
        returntype: Attribute
    ) =>
      parser.verifyCustomOp(
        opGen = factory,
        opName = name,
        resultNames = resNames,
        resultTypes = Seq(returntype),
        operandNames = Seq(operand),
        operandTypes = Seq(itstype)
      )
  )
  // ==----------------------== //
}

case class Norm(
    override val operands: ListType[Value[Attribute]] = ListType(),
    override val successors: ListType[Block] = ListType(),
    override val results: ListType[Value[Attribute]] = ListType(),
    override val regions: ListType[Region] = ListType(),
    override val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends RegisteredOperation(name = "cmath.norm") {

  override def custom_print(
      p: Printer
  ): String = {
    val oper = p.printValue(operands(0))
    val operType = operands(0).typ.custom_print

    s"${name} (${oper} : ${operType}) => ${p.printAttribute(results(0).typ)}"
  }

  override def custom_verify(): Unit = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    dictionaryProperties.size,
    dictionaryAttributes.size
  ) match {
    case (1, 0, 1, 0, 0, 0) =>
    case _ =>
      throw new VerifyException(
        "Norm Operation must only contain 1 operand of 'complex' type, and 1 result of 'f32' or 'f64'."
      )
  }
}

///////////////////
// MUL OPERATION //
///////////////////

object Mul extends DialectOperation {
  override def name: String = "cmath.mul"
  override def factory: FactoryType = Mul.apply

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      resNames: Seq[String],
      parser: Parser
  ): P[Operation] = P(
    "(" ~ ValueId ~ ":" ~ ComplexType.parse ~ "," ~ ValueId ~ ":" ~ ComplexType.parse ~ ")" ~ "=>" ~ ComplexType.parse
  ).map(
    (
        operand1: String,
        itstype1: Attribute,
        operand2: String,
        itstype2: Attribute,
        returntype: Attribute
    ) =>
      parser.verifyCustomOp(
        opGen = factory,
        opName = name,
        resultNames = resNames,
        resultTypes = Seq(returntype),
        operandNames = Seq(operand1, operand2),
        operandTypes = Seq(itstype1, itstype2)
      )
  )
  // ==----------------------== //
}

case class Mul(
    override val operands: ListType[Value[Attribute]] = ListType(),
    override val successors: ListType[Block] = ListType(),
    override val results: ListType[Value[Attribute]] = ListType(),
    override val regions: ListType[Region] = ListType(),
    override val dictionaryProperties: DictType[String, Attribute] =
      DictType.empty[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute] =
      DictType.empty[String, Attribute]
) extends RegisteredOperation(name = "cmath.mul") {

  override def custom_print(
      p: Printer
  ): String = {
    val oper = p.printValue(operands(0))
    val operType = operands(0).typ.custom_print
    val oper1 = p.printValue(operands(1))
    val operType1 = operands(1).typ.custom_print

    s"${name} (${oper} : ${operType}, ${oper1} : ${operType1}) => ${results(0).typ.custom_print}"
  }

  override def custom_verify(): Unit = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    dictionaryProperties.size,
    dictionaryAttributes.size
  ) match {
    case (2, 0, 1, 0, 0, 0) =>
    case _ =>
      throw new VerifyException(
        "Mul Operation must only contain 2 operands and 1 result of 'complex' type."
      )
  }
}

///////////
// CMATH //
///////////

val CMath: Dialect =
  new Dialect(
    operations = Seq(Norm, Mul),
    attributes = Seq(ComplexType)
  )

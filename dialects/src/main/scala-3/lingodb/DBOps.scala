package scair.dialects.LingoDB.DBOps

import fastparse.*
import scair.AttrParser
import scair.EnumAttr.I64EnumAttr
import scair.EnumAttr.I64EnumAttrCase
import scair.Parser
import scair.Parser.ValueId
import scair.Parser.whitespace
import scair.dialects.builtin.*
import scair.exceptions.VerifyException
import scair.ir.*

import scala.collection.immutable
import scala.collection.mutable
import scala.math.max

// ==---== //
//  Enums
// ==---== //

abstract class DB_DateUnit_Case(override val symbol: String)
    extends I64EnumAttrCase(symbol)

abstract class DB_IntervalUnit_Case(override val symbol: String)
    extends I64EnumAttrCase(symbol)

abstract class DB_CmpPredicate_Case(override val symbol: String)
    extends I64EnumAttrCase(symbol)

object DB_DateUnit_DAY extends DB_DateUnit_Case("day")
object DB_DateUnit_MS extends DB_DateUnit_Case("millisecond")

object DB_DateUnitAttr
    extends I64EnumAttr(
      "DateUnitAttr",
      Seq(
        DB_DateUnit_DAY,
        DB_DateUnit_MS
      )
    )

object DB_IntervalUnit_MONTH extends DB_IntervalUnit_Case("months")
object DB_IntervalUnit_DAYTIME extends DB_IntervalUnit_Case("daytime")

object DB_IntervalUnitAttr
    extends I64EnumAttr(
      "IntervalUnitAttr",
      Seq(
        DB_IntervalUnit_MONTH,
        DB_IntervalUnit_DAYTIME
      )
    )

object DB_CMP_P_EQ extends DB_CmpPredicate_Case("eq");
object DB_CMP_P_NEQ extends DB_CmpPredicate_Case("neq");
object DB_CMP_P_LT extends DB_CmpPredicate_Case("lt");
object DB_CMP_P_LTE extends DB_CmpPredicate_Case("lte");
object DB_CMP_P_GT extends DB_CmpPredicate_Case("gt");
object DB_CMP_P_GTE extends DB_CmpPredicate_Case("gte");
object DB_CMP_P_ISA extends DB_CmpPredicate_Case("isa");

object DB_CmpPredicateAttr
    extends I64EnumAttr(
      "DBCmpPredicateAttr",
      Seq(
        DB_CMP_P_EQ,
        DB_CMP_P_NEQ,
        DB_CMP_P_LT,
        DB_CMP_P_LTE,
        DB_CMP_P_GT,
        DB_CMP_P_GTE,
        DB_CMP_P_ISA
      )
    )

///////////
// TYPES //
///////////

// ==--------== //
//   CharType   //
// ==--------== //

object DB_CharType extends AttributeObject {
  override def name: String = "db.char"
  override def factory = DB_CharType.apply
}

case class DB_CharType(val typ: Seq[Attribute])
    extends ParametrizedAttribute(
      name = "db.char",
      parameters = typ
    )
    with TypeAttribute {

  override def custom_verify(): Unit = {
    if (typ.length != 1) {
      throw new VerifyException(
        "TupleStream Tuple must contain 1 elements only."
      )
    } else
      typ(0) match {
        case _: IntData =>
        case _ =>
          throw new VerifyException("CharType type must be IntData")
      }
  }
}

// ==--------== //
//   DateType   //
// ==--------== //

object DB_DateType extends AttributeObject {
  override def name: String = "db.date"
  override def parse[$: P](parser: AttrParser): P[Attribute] =
    P("<" ~ DB_DateUnitAttr.caseParser ~ ">")
      .map((x: Attribute) => DB_DateType(x.asInstanceOf[DB_DateUnit_Case]))
}

case class DB_DateType(val unit: DB_DateUnit_Case)
    extends ParametrizedAttribute(
      name = "db.date",
      parameters = Seq(unit)
    )
    with TypeAttribute

// ==------------== //
//   IntervalType   //
// ==------------== //

object DB_IntervalType extends AttributeObject {
  override def name: String = "db.interval"
  override def parse[$: P](parser: AttrParser): P[Attribute] =
    P("<" ~ DB_IntervalUnitAttr.caseParser ~ ">").map(DB_IntervalType(_))
}

case class DB_IntervalType(val unit: Attribute)
    extends ParametrizedAttribute(
      name = "db.interval",
      parameters = Seq(unit)
    )
    with TypeAttribute {

  // override def custom_verify(): Unit = {
  //   if (typ.length != 1) {
  //     throw new VerifyException("TupleStream Tuple must contain 1 elements only.")
  //   } else
  //     typ(0) match {
  //       case _: IntData =>
  //       case _ =>
  //         throw new VerifyException("CharType type must be IntData")
  //     }
  // }
}

// ==-----------== //
//   DecimalType   //
// ==-----------== //

object DB_DecimalType extends AttributeObject {
  override def name: String = "db.decimal"
  override def factory = DB_DecimalType.apply
}

case class DB_DecimalType(val typ: Seq[Attribute])
    extends ParametrizedAttribute(
      name = "db.decimal",
      parameters = typ
    )
    with TypeAttribute {

  override def custom_verify(): Unit = {
    if (typ.length != 2) {
      throw new VerifyException(
        "TupleStream Tuple must contain exactly 2 elements."
      )
    } else {
      typ(0) match {
        case _: IntData =>
        case _ =>
          throw new VerifyException(
            "DB_DecimalType type must be (IntData, IntData)"
          )
      }
      typ(1) match {
        case _: IntData =>
        case _ =>
          throw new VerifyException(
            "DB_DecimalType type must be (IntData, IntData)"
          )
      }
    }
  }
}

// ==-----------== //
//   StringType   //
// ==-----------== //

object DB_StringType extends AttributeObject {
  override def name: String = "db.string"
  override def factory = DB_StringType.apply
}

case class DB_StringType(val typ: Seq[Attribute])
    extends ParametrizedAttribute(
      name = "db.string",
      parameters = typ
    )
    with TypeAttribute {

  override def custom_verify(): Unit = {
    if (typ.length > 1) {
      throw new VerifyException(
        "TupleStream Tuple must contain at most 1 element."
      )
    } else if (typ.length == 1)
      typ(0) match {
        case _: StringData =>
        case _ =>
          throw new VerifyException("DB_DecimalType type must be StringData")
      }
  }
}

////////////////
// OPERATIONS //
////////////////

// ==----------== //
//   ConstantOp   //
// ==----------== //

object DB_ConstantOp extends OperationObject {
  override def name: String = "db.constant"
  override def factory = DB_ConstantOp.apply

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      resNames: Seq[String],
      parser: Parser
  ): P[Operation] = P(
    "(" ~ parser.Type ~ ")" ~ ":" ~ parser.Type
      ~ parser.DictionaryAttribute.?.map(Parser.optionlessSeq)
  ).map(
    (
        x: Attribute,
        y: Attribute,
        z: Seq[(String, Attribute)]
    ) =>
      parser.verifyCustomOp(
        opGen = factory,
        opName = name,
        resultNames = resNames,
        resultTypes = Seq(y),
        dictAttrs = z :+ ("value", x)
      )
  )
  // ==----------------------== //
}

case class DB_ConstantOp(
    override val operands: ListType[Value[Attribute]],
    override val successors: ListType[Block],
    override val results: ListType[Value[Attribute]],
    override val regions: ListType[Region],
    override val dictionaryProperties: DictType[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute]
) extends RegisteredOperation(name = "db.constant") {

  override def custom_verify(): Unit = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    dictionaryProperties.size
  ) match {
    case (0, 0, 1, 0, 0) =>
    case _ =>
      throw new VerifyException(
        "DB_ConstantOp Operation must contain only 2 dictionary attributes."
      )
  }
}

// ==----------== //
//   CompareOp   //
// ==----------== //

object DB_CmpOp extends OperationObject {
  override def name: String = "db.compare"
  override def factory = DB_CmpOp.apply

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      resNames: Seq[String],
      parser: Parser
  ): P[Operation] = P(
    DB_CmpPredicateAttr.caseParser ~ ValueId ~ ":" ~ parser.Type ~ "," ~ ValueId ~ ":" ~ parser.Type
      ~ parser.DictionaryAttribute.?.map(Parser.optionlessSeq)
  ).map(
    (
        x: Attribute,
        left: String,
        leftType: Attribute,
        right: String,
        rightType: Attribute,
        z: Seq[(String, Attribute)]
    ) =>
      parser.verifyCustomOp(
        opGen = factory,
        opName = name,
        resultNames = resNames,
        resultTypes = Seq(I1),
        operandNames = Seq(left, right),
        operandTypes = Seq(leftType, rightType),
        dictAttrs = z :+ ("predicate", x)
      )
  )
  // ==----------------------== //
}

case class DB_CmpOp(
    override val operands: ListType[Value[Attribute]],
    override val successors: ListType[Block],
    override val results: ListType[Value[Attribute]],
    override val regions: ListType[Region],
    override val dictionaryProperties: DictType[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute]
) extends RegisteredOperation(name = "db.compare") {

  override def custom_verify(): Unit = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    dictionaryProperties.size
  ) match {
    case (2, 0, 0, 0, 0) =>
      (operands(0).typ == operands(1).typ) match {
        case true =>
        case false =>
          throw new VerifyException(
            "In order to be compared, operands' types must match!"
          )
      }
    case _ =>
      throw new VerifyException(
        "DB_CmpOp Operation must contain only 2 operands."
      )
  }
}

// ==-----== //
//   MulOp   //
// ==-----== //

object DB_MulOp extends OperationObject {
  override def name: String = "db.mul"
  override def factory = DB_MulOp.apply

  // ==--- Custom Parsing ---== //
  private def inferRetType(opLeft: Attribute, opRight: Attribute) = {
    (opLeft.getClass == opRight.getClass) match {
      case true =>
        opLeft match {
          case x: IntegerAttr => Seq(opLeft)
          case y: FloatAttr   => Seq(opLeft)
          case z: DB_DecimalType =>
            Seq(
              DB_DecimalType(
                Seq(
                  new IntegerAttr(
                    IntData(
                      opLeft
                        .asInstanceOf[DB_DecimalType]
                        .typ(0)
                        .asInstanceOf[IntegerAttr]
                        .value
                        .value
                        +
                          opRight
                            .asInstanceOf[DB_DecimalType]
                            .typ(0)
                            .asInstanceOf[IntegerAttr]
                            .value
                            .value
                    )
                  ),
                  new IntegerAttr(
                    IntData(
                      opLeft
                        .asInstanceOf[DB_DecimalType]
                        .typ(1)
                        .asInstanceOf[IntegerAttr]
                        .value
                        .value
                        +
                          opRight
                            .asInstanceOf[DB_DecimalType]
                            .typ(1)
                            .asInstanceOf[IntegerAttr]
                            .value
                            .value
                    )
                  )
                )
              )
            )
          case _ =>
            throw new Exception(
              "Operand types for the MulOp must be of type IntegerAttr, FloatAttr or DB_DecimalType."
            )
        }
      case false =>
        throw new Exception("Operand types for the MulOp must be the same.")
    }
  }

  override def parse[$: P](
      resNames: Seq[String],
      parser: Parser
  ): P[Operation] = P(
    ValueId ~ ":" ~ parser.Type ~ "," ~ ValueId ~ ":" ~ parser.Type
      ~ parser.DictionaryAttribute.?.map(Parser.optionlessSeq)
  ).map(
    (
        left: String,
        leftType: Attribute,
        right: String,
        rightType: Attribute,
        z: Seq[(String, Attribute)]
    ) =>
      parser.verifyCustomOp(
        opGen = factory,
        opName = name,
        resultNames = resNames,
        resultTypes = inferRetType(leftType, rightType),
        operandNames = Seq(left, right),
        operandTypes = Seq(leftType, rightType),
        dictAttrs = z
      )
  )
  // ==----------------------== //
}

case class DB_MulOp(
    override val operands: ListType[Value[Attribute]],
    override val successors: ListType[Block],
    override val results: ListType[Value[Attribute]],
    override val regions: ListType[Region],
    override val dictionaryProperties: DictType[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute]
) extends RegisteredOperation(name = "db.mul") {

  override def custom_verify(): Unit = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    dictionaryProperties.size
  ) match {
    case (2, 0, 1, 0, 0) =>
      (operands(0).typ == operands(1).typ) match {
        case true =>
        case false =>
          throw new VerifyException(
            "In order to be multiplied, operands' types must match!"
          )
      }
    case _ =>
      throw new VerifyException(
        "DB_MulOp Operation must contain only 2 operands."
      )
  }
}

// ==-----== //
//   DivOp   //
// ==-----== //

object DB_DivOp extends OperationObject {
  override def name: String = "db.div"
  override def factory = DB_DivOp.apply

  // ==--- Custom Parsing ---== //
  private def inferRetType(opLeft: Attribute, opRight: Attribute) = {
    (opLeft.getClass == opRight.getClass) match {
      case true =>
        opLeft match {
          case x: IntegerAttr => Seq(opLeft)
          case y: FloatAttr   => Seq(opLeft)
          case z: DB_DecimalType =>
            val opLeft0 =
              opLeft
                .asInstanceOf[DB_DecimalType]
                .typ(0)
                .asInstanceOf[IntegerAttr]
                .value
                .value
            val opLeft1 =
              opLeft
                .asInstanceOf[DB_DecimalType]
                .typ(1)
                .asInstanceOf[IntegerAttr]
                .value
                .value
            val opRight0 =
              opRight
                .asInstanceOf[DB_DecimalType]
                .typ(0)
                .asInstanceOf[IntegerAttr]
                .value
                .value
            val opRight1 =
              opRight
                .asInstanceOf[DB_DecimalType]
                .typ(1)
                .asInstanceOf[IntegerAttr]
                .value
                .value
            Seq(
              DB_DecimalType(
                Seq(
                  new IntegerAttr(
                    IntData(
                      opLeft0 - opLeft1 + opRight1
                        + max(opLeft1 + opRight0, 6)
                    )
                  ),
                  new IntegerAttr(
                    IntData(max(opLeft1 + opRight0, 6))
                  )
                )
              )
            )
          case _ =>
            throw new Exception(
              "Operand types for the DivOp must be of type IntegerAttr, FloatAttr or DB_DecimalType."
            )
        }
      case false =>
        throw new Exception("Operand types for the MulOp must be the same.")
    }
  }

  override def parse[$: P](
      resNames: Seq[String],
      parser: Parser
  ): P[Operation] = P(
    ValueId ~ ":" ~ parser.Type ~ "," ~ ValueId ~ ":" ~ parser.Type
      ~ parser.DictionaryAttribute.?.map(Parser.optionlessSeq)
  ).map(
    (
        left: String,
        leftType: Attribute,
        right: String,
        rightType: Attribute,
        z: Seq[(String, Attribute)]
    ) =>
      parser.verifyCustomOp(
        opGen = factory,
        opName = name,
        operandNames = Seq(left, right),
        operandTypes = Seq(leftType, rightType),
        resultNames = resNames,
        resultTypes = inferRetType(leftType, rightType),
        dictAttrs = z
      )
  )
  // ==----------------------== //
}

case class DB_DivOp(
    override val operands: ListType[Value[Attribute]],
    override val successors: ListType[Block],
    override val results: ListType[Value[Attribute]],
    override val regions: ListType[Region],
    override val dictionaryProperties: DictType[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute]
) extends RegisteredOperation(name = "db.div") {

  override def custom_verify(): Unit = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    dictionaryProperties.size
  ) match {
    case (2, 0, 1, 0, 0) =>
      (operands(0).typ == operands(1).typ) match {
        case true =>
        case false =>
          throw new VerifyException(
            "In order to be divided, operands' types must match!"
          )
      }
    case _ =>
      throw new VerifyException(
        "DB_DivOp Operation must contain only 2 operands."
      )
  }
}

// ==-----== //
//   AddOp   //
// ==-----== //

object DB_AddOp extends OperationObject {
  override def name: String = "db.add"
  override def factory = DB_AddOp.apply

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      resNames: Seq[String],
      parser: Parser
  ): P[Operation] = P(
    ValueId ~ ":" ~ parser.Type ~ "," ~ ValueId ~ ":" ~ parser.Type
      ~ parser.DictionaryAttribute.?.map(Parser.optionlessSeq)
  ).map(
    (
        left: String,
        leftType: Attribute,
        right: String,
        rightType: Attribute,
        z: Seq[(String, Attribute)]
    ) =>
      parser.verifyCustomOp(
        opGen = factory,
        opName = name,
        resultNames = resNames,
        resultTypes = Seq(leftType),
        operandNames = Seq(left, right),
        operandTypes = Seq(leftType, rightType),
        dictAttrs = z
      )
  )
  // ==----------------------== //
}

case class DB_AddOp(
    override val operands: ListType[Value[Attribute]],
    override val successors: ListType[Block],
    override val results: ListType[Value[Attribute]],
    override val regions: ListType[Region],
    override val dictionaryProperties: DictType[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute]
) extends RegisteredOperation(name = "db.add") {

  override def custom_verify(): Unit = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    dictionaryProperties.size
  ) match {
    case (2, 0, 1, 0, 0) =>
      (operands(0).typ == operands(1).typ) match {
        case true =>
        case false =>
          throw new VerifyException(
            "In order to be added, operands' types must match!"
          )
      }
    case _ =>
      throw new VerifyException(
        "DB_AddOp Operation must contain only 2 operands."
      )
  }
}

// ==-----== //
//   SubOp   //
// ==-----== //

object DB_SubOp extends OperationObject {
  override def name: String = "db.sub"
  override def factory = DB_SubOp.apply

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      resNames: Seq[String],
      parser: Parser
  ): P[Operation] = P(
    ValueId ~ ":" ~ parser.Type ~ "," ~ ValueId ~ ":" ~ parser.Type
      ~ parser.DictionaryAttribute.?.map(Parser.optionlessSeq)
  ).map(
    (
        left: String,
        leftType: Attribute,
        right: String,
        rightType: Attribute,
        z: Seq[(String, Attribute)]
    ) =>
      parser.verifyCustomOp(
        opGen = factory,
        opName = name,
        resultNames = resNames,
        resultTypes = Seq(leftType),
        operandNames = Seq(left, right),
        operandTypes = Seq(leftType, rightType),
        dictAttrs = z
      )
  )
  // ==----------------------== //
}

case class DB_SubOp(
    override val operands: ListType[Value[Attribute]],
    override val successors: ListType[Block],
    override val results: ListType[Value[Attribute]],
    override val regions: ListType[Region],
    override val dictionaryProperties: DictType[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute]
) extends RegisteredOperation(name = "db.sub") {

  override def custom_verify(): Unit = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    dictionaryProperties.size
  ) match {
    case (2, 0, 1, 0, 0) =>
      (operands(0).typ == operands(1).typ) match {
        case true =>
        case false =>
          throw new VerifyException(
            "In order to carry out substitution, operands' types must match!"
          )
      }
    case _ =>
      throw new VerifyException(
        "DB_SubOp Operation must contain only 2 operands."
      )
  }
}

// ==-----== //
//   CastOp   //
// ==-----== //

object CastOp extends OperationObject {
  override def name: String = "db.cast"
  override def factory = CastOp.apply

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      resNames: Seq[String],
      parser: Parser
  ): P[Operation] = P(
    ValueId ~ ":" ~ parser.Type ~ "->" ~ parser.Type.rep
      ~ parser.DictionaryAttribute.?.map(Parser.optionlessSeq)
  ).map(
    (
        operand: String,
        opType: Attribute,
        resTypes: Seq[Attribute],
        z: Seq[(String, Attribute)]
    ) =>
      parser.verifyCustomOp(
        opGen = factory,
        opName = name,
        resultNames = resNames,
        resultTypes = resTypes,
        operandNames = Seq(operand),
        operandTypes = Seq(opType),
        dictAttrs = z
      )
  )
  // ==----------------------== //
}

case class CastOp(
    override val operands: ListType[Value[Attribute]],
    override val successors: ListType[Block],
    override val results: ListType[Value[Attribute]],
    override val regions: ListType[Region],
    override val dictionaryProperties: DictType[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute]
) extends RegisteredOperation(name = "db.cast") {

  override def custom_verify(): Unit = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    dictionaryProperties.size
  ) match {
    case (1, 0, 1, 0, 0) =>
    case _ =>
      throw new VerifyException(
        "CastOp Operation must contain only 1 operand and result."
      )
  }
}

val DBOps: Dialect =
  new Dialect(
    operations = Seq(
      DB_ConstantOp,
      DB_CmpOp,
      DB_MulOp,
      DB_DivOp,
      DB_SubOp,
      DB_AddOp,
      CastOp
    ),
    attributes = Seq(DB_CharType, DB_DateType, DB_DecimalType, DB_StringType)
  )

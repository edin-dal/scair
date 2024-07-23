package scair.dialects.LingoDB.DBOps

import fastparse._
import scair.EnumAttr.{I64EnumAttrCase, I64EnumAttr}
import scair.dialects.builtin._
import scala.collection.immutable
import scair.dialects.irdl.{Operand, OpResult}
import scair.Parser.{whitespace, ValueId, Type, DictionaryAttribute}
import scair.{
  RegisteredOperation,
  Region,
  Block,
  Value,
  Attribute,
  TypeAttribute,
  ParametrizedAttribute,
  DialectAttribute,
  DialectOperation,
  Dialect,
  Parser,
  Operation,
  AttrParser
}

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

object DB_CharType extends DialectAttribute {
  override def name: String = "db.char"
  override def factory = DB_CharType.apply
}

case class DB_CharType(val typ: Seq[Attribute])
    extends ParametrizedAttribute(
      name = "db.char",
      parameters = typ
    )
    with TypeAttribute {

  override def verify(): Unit = {
    if (typ.length != 1) {
      throw new Exception("TupleStream Tuple must contain 1 elements only.")
    } else
      typ(0) match {
        case _: IntAttr =>
        case _ =>
          throw new Exception("CharType type must be IntAttr")
      }
  }
}

// ==--------== //
//   DateType   //
// ==--------== //

object DB_DateType extends DialectAttribute {
  override def name: String = "db.date"
  override def parse[$: P]: P[Attribute] =
    P("<" ~ DB_DateUnitAttr.caseParser ~ ">").map(DB_DateType(_))
}

case class DB_DateType(val unit: Attribute)
    extends ParametrizedAttribute(
      name = "db.date",
      parameters = Seq(unit)
    )
    with TypeAttribute {

  // override def verify(): Unit = {
  //   if (typ.length != 1) {
  //     throw new Exception("TupleStream Tuple must contain 1 elements only.")
  //   } else
  //     typ(0) match {
  //       case _: IntAttr =>
  //       case _ =>
  //         throw new Exception("CharType type must be IntAttr")
  //     }
  // }
}

// ==------------== //
//   IntervalType   //
// ==------------== //

object DB_IntervalType extends DialectAttribute {
  override def name: String = "db.interval"
  override def parse[$: P]: P[Attribute] =
    P("<" ~ DB_IntervalUnitAttr.caseParser ~ ">").map(DB_IntervalType(_))
}

case class DB_IntervalType(val unit: Attribute)
    extends ParametrizedAttribute(
      name = "db.interval",
      parameters = Seq(unit)
    )
    with TypeAttribute {

  // override def verify(): Unit = {
  //   if (typ.length != 1) {
  //     throw new Exception("TupleStream Tuple must contain 1 elements only.")
  //   } else
  //     typ(0) match {
  //       case _: IntAttr =>
  //       case _ =>
  //         throw new Exception("CharType type must be IntAttr")
  //     }
  // }
}

// ==-----------== //
//   DecimalType   //
// ==-----------== //

object DB_DecimalType extends DialectAttribute {
  override def name: String = "db.decimal"
  override def factory = DB_DecimalType.apply
}

case class DB_DecimalType(val typ: Seq[Attribute])
    extends ParametrizedAttribute(
      name = "db.decimal",
      parameters = typ
    )
    with TypeAttribute {

  override def verify(): Unit = {
    if (typ.length != 2) {
      throw new Exception("TupleStream Tuple must contain exactly 2 elements.")
    } else
      (typ(0), typ(1)) match {
        case _: (IntAttr, IntAttr) =>
        case _ =>
          throw new Exception("DB_DecimalType type must be (IntAttr, IntAttr)")
      }
  }
}

// ==-----------== //
//   StringType   //
// ==-----------== //

object DB_StringType extends DialectAttribute {
  override def name: String = "db.string"
  override def factory = DB_StringType.apply
}

case class DB_StringType(val typ: Seq[Attribute])
    extends ParametrizedAttribute(
      name = "db.string",
      parameters = typ
    )
    with TypeAttribute {

  override def verify(): Unit = {
    if (typ.length > 1) {
      throw new Exception("TupleStream Tuple must contain at most 1 element.")
    } else if (typ.length == 1)
      typ(0) match {
        case _: StringAttribute =>
        case _ =>
          throw new Exception("DB_DecimalType type must be StringAttribute")
      }
  }
}

////////////////
// OPERATIONS //
////////////////

// ==----------== //
//   ConstantOp   //
// ==----------== //

object DB_ConstantOp extends DialectOperation {
  override def name: String = "db.constant"
  override def factory = DB_ConstantOp.apply

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      resNames: Seq[String],
      parser: Parser
  ): P[Operation] = P(
    "(" ~ Type ~ ")" ~ ":" ~ Type
      ~ DictionaryAttribute.?.map(Parser.optionlessSeq)
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
    override val operands: collection.mutable.ArrayBuffer[Value[Attribute]],
    override val successors: collection.mutable.ArrayBuffer[Block],
    override val results: Seq[Value[Attribute]],
    override val regions: Seq[Region],
    override val dictionaryProperties: immutable.Map[String, Attribute],
    override val dictionaryAttributes: immutable.Map[String, Attribute]
) extends RegisteredOperation(name = "db.constant") {

  override def verify(): Unit = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    dictionaryProperties.size,
    dictionaryAttributes.size
  ) match {
    case (0, 0, 0, 0, 0, _) =>
      for ((x, y) <- dictionaryAttributes) yield y.verify()
    case _ =>
      throw new Exception(
        "DB_ConstantOp Operation must contain only 2 dictionary attributes."
      )
  }
}

// ==----------== //
//   CompareOp   //
// ==----------== //

object DB_CmpOp extends DialectOperation {
  override def name: String = "db.compare"
  override def factory = DB_CmpOp.apply

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      resNames: Seq[String],
      parser: Parser
  ): P[Operation] = P(
    DB_CmpPredicateAttr.caseParser ~ ValueId ~ ":" ~ Type ~ "," ~ ValueId ~ ":" ~ Type
      ~ DictionaryAttribute.?.map(Parser.optionlessSeq)
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
        resultTypes = Seq(IntegerType(8, Signless)),
        operandNames = Seq(left, right),
        operandTypes = Seq(leftType, rightType),
        dictAttrs = z :+ ("predicate", x)
      )
  )
  // ==----------------------== //
}

case class DB_CmpOp(
    override val operands: collection.mutable.ArrayBuffer[Value[Attribute]],
    override val successors: collection.mutable.ArrayBuffer[Block],
    override val results: Seq[Value[Attribute]],
    override val regions: Seq[Region],
    override val dictionaryProperties: immutable.Map[String, Attribute],
    override val dictionaryAttributes: immutable.Map[String, Attribute]
) extends RegisteredOperation(name = "db.compare") {

  override def verify(): Unit = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    dictionaryProperties.size
  ) match {
    case (2, 0, 0, 0, 0) =>
      operands(0).typ.verify()
      operands(1).typ.verify()
      (operands(0).typ == operands(1).typ) match {
        case true =>
        case false =>
          throw new Exception(
            "In order to be compared, operands' types must match!"
          )
      }
    case _ =>
      throw new Exception(
        "DB_CmpOp Operation must contain only 2 operands."
      )
  }
}

// ==-----== //
//   AddOp   //
// ==-----== //

object DB_AddOp extends DialectOperation {
  override def name: String = "db.add"
  override def factory = DB_AddOp.apply

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      resNames: Seq[String],
      parser: Parser
  ): P[Operation] = P(
    ValueId ~ ":" ~ Type ~ "," ~ ValueId ~ ":" ~ Type
      ~ DictionaryAttribute.?.map(Parser.optionlessSeq)
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
        resultTypes = Seq(I64),
        operandNames = Seq(left, right),
        operandTypes = Seq(leftType, rightType),
        dictAttrs = z
      )
  )
  // ==----------------------== //
}

case class DB_AddOp(
    override val operands: collection.mutable.ArrayBuffer[Value[Attribute]],
    override val successors: collection.mutable.ArrayBuffer[Block],
    override val results: Seq[Value[Attribute]],
    override val regions: Seq[Region],
    override val dictionaryProperties: immutable.Map[String, Attribute],
    override val dictionaryAttributes: immutable.Map[String, Attribute]
) extends RegisteredOperation(name = "db.add") {

  override def verify(): Unit = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    dictionaryProperties.size
  ) match {
    case (2, 0, 0, 0, 0) =>
      operands(0).typ.verify()
      operands(1).typ.verify()
      (operands(0).typ == operands(1).typ) match {
        case true =>
        case false =>
          throw new Exception(
            "In order to be compared, operands' types must match!"
          )
      }
    case _ =>
      throw new Exception(
        "DB_AddOp Operation must contain only 2 operands."
      )
  }
}

// ==-----== //
//   MulOp   //
// ==-----== //

object DB_MulOp extends DialectOperation {
  override def name: String = "db.mul"
  override def factory = DB_MulOp.apply

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      resNames: Seq[String],
      parser: Parser
  ): P[Operation] = P(
    ValueId ~ ":" ~ Type ~ "," ~ ValueId ~ ":" ~ Type
      ~ DictionaryAttribute.?.map(Parser.optionlessSeq)
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
        resultTypes = Seq(I64),
        operandNames = Seq(left, right),
        operandTypes = Seq(leftType, rightType),
        dictAttrs = z
      )
  )
  // ==----------------------== //
}

case class DB_MulOp(
    override val operands: collection.mutable.ArrayBuffer[Value[Attribute]],
    override val successors: collection.mutable.ArrayBuffer[Block],
    override val results: Seq[Value[Attribute]],
    override val regions: Seq[Region],
    override val dictionaryProperties: immutable.Map[String, Attribute],
    override val dictionaryAttributes: immutable.Map[String, Attribute]
) extends RegisteredOperation(name = "db.mul") {

  override def verify(): Unit = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    dictionaryProperties.size
  ) match {
    case (2, 0, 0, 0, 0) =>
      operands(0).typ.verify()
      operands(1).typ.verify()
      (operands(0).typ == operands(1).typ) match {
        case true =>
        case false =>
          throw new Exception(
            "In order to be compared, operands' types must match!"
          )
      }
    case _ =>
      throw new Exception(
        "DB_MulOp Operation must contain only 2 operands."
      )
  }
}

// ==-----== //
//   SubOp   //
// ==-----== //

object DB_SubOp extends DialectOperation {
  override def name: String = "db.sub"
  override def factory = DB_SubOp.apply

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      resNames: Seq[String],
      parser: Parser
  ): P[Operation] = P(
    ValueId ~ ":" ~ Type ~ "," ~ ValueId ~ ":" ~ Type
      ~ DictionaryAttribute.?.map(Parser.optionlessSeq)
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
        resultTypes = Seq(I64),
        operandNames = Seq(left, right),
        operandTypes = Seq(leftType, rightType),
        dictAttrs = z
      )
  )
  // ==----------------------== //
}

case class DB_SubOp(
    override val operands: collection.mutable.ArrayBuffer[Value[Attribute]],
    override val successors: collection.mutable.ArrayBuffer[Block],
    override val results: Seq[Value[Attribute]],
    override val regions: Seq[Region],
    override val dictionaryProperties: immutable.Map[String, Attribute],
    override val dictionaryAttributes: immutable.Map[String, Attribute]
) extends RegisteredOperation(name = "db.sub") {

  override def verify(): Unit = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    dictionaryProperties.size
  ) match {
    case (2, 0, 0, 0, 0) =>
      operands(0).typ.verify()
      operands(1).typ.verify()
      (operands(0).typ == operands(1).typ) match {
        case true =>
        case false =>
          throw new Exception(
            "In order to be compared, operands' types must match!"
          )
      }
    case _ =>
      throw new Exception(
        "DB_SubOp Operation must contain only 2 operands."
      )
  }
}

val DBOps: Dialect =
  new Dialect(
    operations = Seq(DB_ConstantOp, DB_CmpOp, DB_AddOp, DB_MulOp, DB_SubOp),
    attributes = Seq(DB_CharType, DB_DateType, DB_DecimalType, DB_StringType)
  )

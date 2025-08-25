package scair.dialects.LingoDB.DBOps

import fastparse.*
import scair.AttrParser
import scair.EnumAttr.I64EnumAttr
import scair.EnumAttr.I64EnumAttrCase
import scair.Parser
import scair.Parser.*
import scair.Printer
import scair.dialects.builtin.*
import scair.ir.*

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

object DB_CharType extends AttributeCompanion {
  override def name: String = "db.char"

  override def parse[$: P](p: AttrParser) =
    P(("<" ~/ p.Attribute.rep(sep = ",") ~ ">").orElse(Seq()))
      .map(DB_CharType(_))

}

case class DB_CharType(val typ: Seq[Attribute])
    extends ParametrizedAttribute
    with TypeAttribute {

  override def name: String = "db.char"
  override def parameters: Seq[Attribute] = typ

  override def custom_verify(): Either[String, Unit] = {
    if (typ.length != 1) {
      Left("TupleStream Tuple must contain 1 elements only.")
    } else
      typ(0) match {
        case _: IntData => Right(())
        case _          => Left("CharType type must be IntData")
      }
  }

  override def custom_print(p: Printer) = {
    p.print("!", name)(using indentLevel = 0)
    if typ.nonEmpty then
      p.printList(
        typ,
        "<",
        ", ",
        ">"
      )
  }

}

// ==--------== //
//   DateType   //
// ==--------== //

object DB_DateType extends AttributeCompanion {
  override def name: String = "db.date"

  override def parse[$: P](parser: AttrParser): P[Attribute] =
    P("<" ~ DB_DateUnitAttr.caseParser ~ ">")
      .map((x: Attribute) => DB_DateType(x.asInstanceOf[DB_DateUnit_Case]))

}

case class DB_DateType(val unit: DB_DateUnit_Case)
    extends ParametrizedAttribute
    with TypeAttribute {
  override def name: String = "db.date"
  override def parameters: Seq[Attribute | Seq[Attribute]] = Seq(unit)
}

// ==------------== //
//   IntervalType   //
// ==------------== //

object DB_IntervalType extends AttributeCompanion {
  override def name: String = "db.interval"

  override def parse[$: P](parser: AttrParser): P[Attribute] =
    P("<" ~ DB_IntervalUnitAttr.caseParser ~ ">").map(DB_IntervalType(_))

}

case class DB_IntervalType(val unit: Attribute)
    extends ParametrizedAttribute
    with TypeAttribute {

  override def name: String = "db.interval"
  override def parameters: Seq[Attribute | Seq[Attribute]] = Seq(unit)

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

  override def custom_print(p: Printer) =
    p.print("!", name)(using indentLevel = 0)
    if parameters.nonEmpty then
      p.printList(
        Seq(unit),
        "<",
        ", ",
        ">"
      )

}

// ==-----------== //
//   DecimalType   //
// ==-----------== //

object DB_DecimalType extends AttributeCompanion {
  override def name: String = "db.decimal"

  override def parse[$: P](p: AttrParser) =
    P(("<" ~/ p.Attribute.rep(sep = ",") ~ ">").orElse(Seq()))
      .map(DB_DecimalType(_))

}

case class DB_DecimalType(val typ: Seq[Attribute])
    extends ParametrizedAttribute
    with TypeAttribute {

  override def name: String = "db.decimal"
  override def parameters: Seq[Attribute | Seq[Attribute]] = typ

  override def custom_verify(): Either[String, Unit] = {
    if (typ.length != 2) {
      Left("TupleStream Tuple must contain exactly 2 elements.")
    } else {
      typ(0) match {
        case _: IntData => Right(())
        case _          =>
          Left("DB_DecimalType type must be (IntData, IntData)")
      }
      typ(1) match {
        case _: IntData => Right(())
        case _          =>
          Left("DB_DecimalType type must be (IntData, IntData)")
      }
    }
  }

  override def custom_print(p: Printer) =
    p.print("!", name)(using indentLevel = 0)
    if typ.nonEmpty then
      p.printList(
        typ,
        "<",
        ", ",
        ">"
      )

}

// ==-----------== //
//   StringType   //
// ==-----------== //

object DB_StringType extends AttributeCompanion {
  override def name: String = "db.string"

  override def parse[$: P](p: AttrParser) =
    P(("<" ~/ p.Type.rep(sep = ",") ~ ">").orElse(Seq())).map(DB_StringType(_))

}

case class DB_StringType(val typ: Seq[Attribute])
    extends ParametrizedAttribute
    with TypeAttribute {

  override def name: String = "db.string"
  override def parameters: Seq[Attribute | Seq[Attribute]] = typ

  override def custom_verify(): Either[String, Unit] = {
    if (typ.length > 1) {
      Left(
        "TupleStream Tuple must contain at most 1 element."
      )
    } else if (typ.length == 1)
      typ(0) match {
        case _: StringData => Right(())
        case _             =>
          Left("DB_DecimalType type must be StringData")
      }
    else Right(())
  }

  override def custom_print(p: Printer) =
    p.print("!", name)(using indentLevel = 0)
    if typ.nonEmpty then
      p.printList(
        typ,
        "<",
        ", ",
        ">"
      )

}

////////////////
// OPERATIONS //
////////////////

// ==----------== //
//   ConstantOp   //
// ==----------== //

object DB_ConstantOp extends OperationCompanion {
  override def name: String = "db.constant"

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      parser: Parser
  ): P[Operation] = P(
    "(" ~ parser.Attribute ~ ")" ~ ":" ~ parser.Type
      ~ parser.OptionalAttributes
  ).map(
    (
        x: Attribute,
        y: Attribute,
        z: Map[String, Attribute]
    ) =>
      parser.generateOperation(
        opName = name,
        resultsTypes = Seq(y),
        attributes = z + ("value" -> x)
      )
  )
  // ==----------------------== //

}

case class DB_ConstantOp(
    override val operands: Seq[Value[Attribute]],
    override val successors: Seq[Block],
    override val results: Seq[Result[Attribute]],
    override val regions: Seq[Region],
    override val properties: Map[String, Attribute],
    override val attributes: DictType[String, Attribute]
) extends BaseOperation(
      name = "db.constant",
      operands,
      successors,
      results,
      regions,
      properties,
      attributes
    ) {

  override def custom_verify(): Either[String, Operation] = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    properties.size
  ) match {
    case (0, 0, 1, 0, 0) => Right(this)
    case _               =>
      Left(
        "DB_ConstantOp Operation must contain only 2 dictionary attributes."
      )
  }

  override def custom_print(printer: Printer)(using indentLevel: Int) = {
    val value =
      attributes.get("value").map(_.toString).getOrElse("")
    val resultType = results.head.typ
    printer.print(s"$name($value) : $resultType")
  }

}

// ==----------== //
//   CompareOp   //
// ==----------== //

object DB_CmpOp extends OperationCompanion {
  override def name: String = "db.compare"

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      parser: Parser
  ): P[Operation] = P(
    DB_CmpPredicateAttr.caseParser ~ ValueId ~ ":" ~ parser.Type ~ "," ~ ValueId ~ ":" ~ parser.Type
      ~ parser.OptionalAttributes
  ).map(
    (
        x: Attribute,
        left: String,
        leftType: Attribute,
        right: String,
        rightType: Attribute,
        z: Map[String, Attribute]
    ) =>
      parser.generateOperation(
        opName = name,
        resultsTypes = Seq(I1),
        operandsNames = Seq(left, right),
        operandsTypes = Seq(leftType, rightType),
        attributes = z + ("predicate" -> x)
      )
  )
  // ==----------------------== //

}

case class DB_CmpOp(
    override val operands: Seq[Value[Attribute]],
    override val successors: Seq[Block],
    override val results: Seq[Result[Attribute]],
    override val regions: Seq[Region],
    override val properties: Map[String, Attribute],
    override val attributes: DictType[String, Attribute]
) extends BaseOperation(
      name = "db.compare",
      operands,
      successors,
      results,
      regions,
      properties,
      attributes
    ) {

  override def custom_verify(): Either[String, Operation] = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    properties.size
  ) match {
    case (2, 0, 0, 0, 0) =>
      (operands(0).typ == operands(1).typ) match {
        case true  => Right(this)
        case false =>
          Left(
            "In order to be compared, operands' types must match!"
          )
      }
    case _ =>
      Left(
        "DB_CmpOp Operation must contain only 2 operands."
      )
  }

  override def custom_print(printer: Printer)(using indentLevel: Int) = {
    printer.print(name)
    printer.print(s" ${attributes("predicate")} ")
    printer.printListF(
      operands,
      printer.printArgument
    )
    printer.print(" : ")
    printer.print(results.head.typ)
  }

}

// ==-----== //
//   MulOp   //
// ==-----== //

object DB_MulOp extends OperationCompanion {
  override def name: String = "db.mul"

  // ==--- Custom Parsing ---== //
  private def inferRetType(opLeft: Attribute, opRight: Attribute) = {
    (opLeft.getClass == opRight.getClass) match {
      case true =>
        opLeft match {
          case x: IntegerAttr    => Seq(opLeft)
          case y: FloatAttr      => Seq(opLeft)
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
      parser: Parser
  ): P[Operation] = P(
    ValueId ~ ":" ~ parser.Type ~ "," ~ ValueId ~ ":" ~ parser.Type
      ~ parser.OptionalAttributes
  ).map(
    (
        left: String,
        leftType: Attribute,
        right: String,
        rightType: Attribute,
        z: Map[String, Attribute]
    ) =>
      parser.generateOperation(
        opName = name,
        resultsTypes = inferRetType(leftType, rightType),
        operandsNames = Seq(left, right),
        operandsTypes = Seq(leftType, rightType),
        attributes = z
      )
  )
  // ==----------------------== //

}

case class DB_MulOp(
    override val operands: Seq[Value[Attribute]],
    override val successors: Seq[Block],
    override val results: Seq[Result[Attribute]],
    override val regions: Seq[Region],
    override val properties: Map[String, Attribute],
    override val attributes: DictType[String, Attribute]
) extends BaseOperation(
      name = "db.mul",
      operands,
      successors,
      results,
      regions,
      properties,
      attributes
    ) {

  override def custom_verify(): Either[String, Operation] = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    properties.size
  ) match {
    case (2, 0, 1, 0, 0) =>
      (operands(0).typ == operands(1).typ) match {
        case true  => Right(this)
        case false =>
          Left(
            "In order to be multiplied, operands' types must match!"
          )
      }
    case _ =>
      Left(
        "DB_MulOp Operation must contain only 2 operands."
      )
  }

  // added code for custom printing

  override def custom_print(printer: Printer)(using indentLevel: Int) = {
    printer.print(name, " ")
    printer.printListF(operands, printer.printArgument)
    printer.print(" : ")
    printer.print(results.head.typ)
  }

}

// ==-----== //
//   DivOp   //
// ==-----== //

object DB_DivOp extends OperationCompanion {
  override def name: String = "db.div"

  // ==--- Custom Parsing ---== //
  private def inferRetType(opLeft: Attribute, opRight: Attribute) = {
    (opLeft.getClass == opRight.getClass) match {
      case true =>
        opLeft match {
          case x: IntegerAttr    => Seq(opLeft)
          case y: FloatAttr      => Seq(opLeft)
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
                        + (opLeft1 + opRight0).max(6)
                    )
                  ),
                  new IntegerAttr(
                    IntData((opLeft1 + opRight0).max(6))
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
      parser: Parser
  ): P[Operation] = P(
    ValueId ~ ":" ~ parser.Type ~ "," ~ ValueId ~ ":" ~ parser.Type
      ~ parser.OptionalAttributes
  ).map(
    (
        left: String,
        leftType: Attribute,
        right: String,
        rightType: Attribute,
        z: Map[String, Attribute]
    ) =>
      parser.generateOperation(
        opName = name,
        operandsNames = Seq(left, right),
        operandsTypes = Seq(leftType, rightType),
        resultsTypes = inferRetType(leftType, rightType),
        attributes = z
      )
  )
  // ==----------------------== //

}

case class DB_DivOp(
    override val operands: Seq[Value[Attribute]],
    override val successors: Seq[Block],
    override val results: Seq[Result[Attribute]],
    override val regions: Seq[Region],
    override val properties: Map[String, Attribute],
    override val attributes: DictType[String, Attribute]
) extends BaseOperation(
      name = "db.div",
      operands,
      successors,
      results,
      regions,
      properties,
      attributes
    ) {

  override def custom_verify(): Either[String, Operation] = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    properties.size
  ) match {
    case (2, 0, 1, 0, 0) =>
      (operands(0).typ == operands(1).typ) match {
        case true  => Right(this)
        case false =>
          Left(
            "In order to be divided, operands' types must match!"
          )
      }
    case _ =>
      Left(
        "DB_DivOp Operation must contain only 2 operands."
      )
  }

  override def custom_print(printer: Printer)(using indentLevel: Int) = {
    printer.print(s"$name ")
    printer.printListF(
      operands,
      printer.printArgument
    )
    printer.print(" : ")
    printer.print(results.head.typ)
  }

}

// ==-----== //
//   AddOp   //
// ==-----== //

object DB_AddOp extends OperationCompanion {
  override def name: String = "db.add"

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      parser: Parser
  ): P[Operation] = P(
    ValueId ~ ":" ~ parser.Type ~ "," ~ ValueId ~ ":" ~ parser.Type
      ~ parser.OptionalAttributes
  ).map(
    (
        left: String,
        leftType: Attribute,
        right: String,
        rightType: Attribute,
        z: Map[String, Attribute]
    ) =>
      parser.generateOperation(
        opName = name,
        resultsTypes = Seq(leftType),
        operandsNames = Seq(left, right),
        operandsTypes = Seq(leftType, rightType),
        attributes = z
      )
  )
  // ==----------------------== //

}

case class DB_AddOp(
    override val operands: Seq[Value[Attribute]],
    override val successors: Seq[Block],
    override val results: Seq[Result[Attribute]],
    override val regions: Seq[Region],
    override val properties: Map[String, Attribute],
    override val attributes: DictType[String, Attribute]
) extends BaseOperation(
      name = "db.add",
      operands,
      successors,
      results,
      regions,
      properties,
      attributes
    ) {

  override def custom_verify(): Either[String, Operation] = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    properties.size
  ) match {
    case (2, 0, 1, 0, 0) =>
      (operands(0).typ == operands(1).typ) match {
        case true  => Right(this)
        case false =>
          Left(
            "In order to be added, operands' types must match!"
          )
      }
    case _ =>
      Left(
        "DB_AddOp Operation must contain only 2 operands."
      )
  }

//added code for custom printing

  override def custom_print(printer: Printer)(using indentLevel: Int) = {
    printer.print(s"$name ")
    printer.printListF(
      operands,
      printer.printArgument
    )
    printer.print(" : ")
    printer.print(results.head.typ)
  }

}

// ==-----== //
//   SubOp   //
// ==-----== //

object DB_SubOp extends OperationCompanion {
  override def name: String = "db.sub"

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      parser: Parser
  ): P[Operation] = P(
    ValueId ~ ":" ~ parser.Type ~ "," ~ ValueId ~ ":" ~ parser.Type
      ~ parser.OptionalAttributes
  ).map(
    (
        left: String,
        leftType: Attribute,
        right: String,
        rightType: Attribute,
        z: Map[String, Attribute]
    ) =>
      parser.generateOperation(
        opName = name,
        resultsTypes = Seq(leftType),
        operandsNames = Seq(left, right),
        operandsTypes = Seq(leftType, rightType),
        attributes = z
      )
  )
  // ==----------------------== //

}

case class DB_SubOp(
    override val operands: Seq[Value[Attribute]],
    override val successors: Seq[Block],
    override val results: Seq[Result[Attribute]],
    override val regions: Seq[Region],
    override val properties: Map[String, Attribute],
    override val attributes: DictType[String, Attribute]
) extends BaseOperation(
      name = "db.sub",
      operands,
      successors,
      results,
      regions,
      properties,
      attributes
    ) {

  override def custom_verify(): Either[String, Operation] = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    properties.size
  ) match {
    case (2, 0, 1, 0, 0) =>
      (operands(0).typ == operands(1).typ) match {
        case true  => Right(this)
        case false =>
          Left(
            "In order to carry out substitution, operands' types must match!"
          )
      }
    case _ =>
      Left(
        "DB_SubOp Operation must contain only 2 operands."
      )
  }

  override def custom_print(printer: Printer)(using indentLevel: Int) = {
    printer.print(s"$name ")
    printer.printListF(
      operands,
      printer.printArgument
    )
    printer.print(" : ")
    printer.print(results.head.typ)
  }

}

// ==-----== //
//   CastOp   //
// ==-----== //

object CastOp extends OperationCompanion {
  override def name: String = "db.cast"

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      parser: Parser
  ): P[Operation] = P(
    ValueId ~ ":" ~ parser.Type ~ "->" ~ parser.Type.rep
      ~ parser.OptionalAttributes
  ).map(
    (
        operand: String,
        opType: Attribute,
        resTypes: Seq[Attribute],
        z: Map[String, Attribute]
    ) =>
      parser.generateOperation(
        opName = name,
        resultsTypes = resTypes,
        operandsNames = Seq(operand),
        operandsTypes = Seq(opType),
        attributes = z
      )
  )
  // ==----------------------== //

}

case class CastOp(
    override val operands: Seq[Value[Attribute]],
    override val successors: Seq[Block],
    override val results: Seq[Result[Attribute]],
    override val regions: Seq[Region],
    override val properties: Map[String, Attribute],
    override val attributes: DictType[String, Attribute]
) extends BaseOperation(
      name = "db.cast",
      operands,
      successors,
      results,
      regions,
      properties,
      attributes
    ) {

  override def custom_verify(): Either[String, Operation] = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    properties.size
  ) match {
    case (1, 0, 1, 0, 0) => Right(this)
    case _               =>
      Left(
        "CastOp Operation must contain only 1 operand and result."
      )
  }

  override def custom_print(printer: Printer)(using indentLevel: Int) = {
    printer.print(s"$name ")
    printer.printArgument(operands.head)
    printer.print(" -> ")
    printer.print(results.head.typ)
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

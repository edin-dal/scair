package scair.dialects.LingoDB.DBOps

import fastparse.*
import scair.AttrParser
import scair.AttrParser.whitespace
import scair.EnumAttr.I64EnumAttr
import scair.EnumAttr.I64EnumAttrCase
import scair.Parser
import scair.Parser.*
import scair.Printer
import scair.clair.macros.DerivedOperation
import scair.clair.macros.summonDialect
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

object DB_CharType extends AttributeCompanion:
  override def name: String = "db.char"

  override def parse[$: P](p: AttrParser) =
    P(("<" ~/ p.Attribute.rep(sep = ",") ~ ">").orElse(Seq()))
      .map(DB_CharType(_))

case class DB_CharType(val typ: Seq[Attribute])
    extends ParametrizedAttribute
    with TypeAttribute:

  override def name: String = "db.char"
  override def parameters: Seq[Attribute] = typ

  override def custom_verify(): Either[String, Unit] =
    if typ.length != 1 then
      Left("TupleStream Tuple must contain 1 elements only.")
    else
      typ(0) match
        case _: IntData => Right(())
        case _          => Left("CharType type must be IntData")

  override def custom_print(p: Printer) =
    p.print("!", name)(using indentLevel = 0)
    if typ.nonEmpty then
      p.printList(
        typ,
        "<",
        ", ",
        ">"
      )

// ==--------== //
//   DateType   //
// ==--------== //

object DB_DateType extends AttributeCompanion:
  override def name: String = "db.date"

  override def parse[$: P](parser: AttrParser): P[Attribute] =
    P("<" ~ DB_DateUnitAttr.caseParser ~ ">")
      .map((x: Attribute) => DB_DateType(x.asInstanceOf[DB_DateUnit_Case]))

case class DB_DateType(val unit: DB_DateUnit_Case)
    extends ParametrizedAttribute
    with TypeAttribute:
  override def name: String = "db.date"
  override def parameters: Seq[Attribute | Seq[Attribute]] = Seq(unit)

// ==------------== //
//   IntervalType   //
// ==------------== //

object DB_IntervalType extends AttributeCompanion:
  override def name: String = "db.interval"

  override def parse[$: P](parser: AttrParser): P[Attribute] =
    P("<" ~ DB_IntervalUnitAttr.caseParser ~ ">").map(DB_IntervalType(_))

case class DB_IntervalType(val unit: Attribute)
    extends ParametrizedAttribute
    with TypeAttribute:

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

// ==-----------== //
//   DecimalType   //
// ==-----------== //

object DB_DecimalType extends AttributeCompanion:
  override def name: String = "db.decimal"

  override def parse[$: P](p: AttrParser) =
    P(("<" ~/ p.Attribute.rep(sep = ",") ~ ">").orElse(Seq()))
      .map(DB_DecimalType(_))

case class DB_DecimalType(val typ: Seq[Attribute])
    extends ParametrizedAttribute
    with TypeAttribute:

  override def name: String = "db.decimal"
  override def parameters: Seq[Attribute | Seq[Attribute]] = typ

  override def custom_verify(): Either[String, Unit] =
    if typ.length != 2 then
      Left("TupleStream Tuple must contain exactly 2 elements.")
    else
      typ(0) match
        case _: IntData => Right(())
        case _          =>
          Left("DB_DecimalType type must be (IntData, IntData)")
      typ(1) match
        case _: IntData => Right(())
        case _          =>
          Left("DB_DecimalType type must be (IntData, IntData)")

  override def custom_print(p: Printer) =
    p.print("!", name)(using indentLevel = 0)
    if typ.nonEmpty then
      p.printList(
        typ,
        "<",
        ", ",
        ">"
      )

// ==-----------== //
//   StringType   //
// ==-----------== //

object DB_StringType extends AttributeCompanion:
  override def name: String = "db.string"

  override def parse[$: P](p: AttrParser) =
    P(("<" ~/ p.Type.rep(sep = ",") ~ ">").orElse(Seq())).map(DB_StringType(_))

case class DB_StringType(val typ: Seq[Attribute])
    extends ParametrizedAttribute
    with TypeAttribute:

  override def name: String = "db.string"
  override def parameters: Seq[Attribute | Seq[Attribute]] = typ

  override def custom_verify(): Either[String, Unit] =
    if typ.length > 1 then
      Left(
        "TupleStream Tuple must contain at most 1 element."
      )
    else if typ.length == 1 then
      typ(0) match
        case _: StringData => Right(())
        case _             =>
          Left("DB_DecimalType type must be StringData")
    else Right(())

  override def custom_print(p: Printer) =
    p.print("!", name)(using indentLevel = 0)
    if typ.nonEmpty then
      p.printList(
        typ,
        "<",
        ", ",
        ">"
      )

////////////////
// OPERATIONS //
////////////////

// ==----------== //
//   ConstantOp   //
// ==----------== //

object DB_ConstantOp:

  def parse[$: P](
      parser: Parser,
      resNames: Seq[String]
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
        opName = "db.constant",
        resultsNames = resNames,
        resultsTypes = Seq(y),
        attributes = z + ("value" -> x)
      )
  )

  // ==----------------------== //

case class DB_ConstantOp(
    value: Attribute,
    result: Result[Attribute]
) extends DerivedOperation["db.constant", DB_ConstantOp]:

  override def custom_print(printer: Printer)(using indentLevel: Int) =
    val resultType = result.typ
    printer.print(s"$name($value) : $resultType")

// ==----------== //
//   CompareOp   //
// ==----------== //

object DB_CmpOp:

  // ==--- Custom Parsing ---== //
  def parse[$: P](
      parser: Parser,
      resNames: Seq[String]
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
        opName = "db.compare",
        resultsNames = resNames,
        resultsTypes = Seq(I1),
        operandsNames = Seq(left, right),
        operandsTypes = Seq(leftType, rightType),
        attributes = z + ("predicate" -> x)
      )
  )

  // ==----------------------== //

case class DB_CmpOp(
    predicate: DB_CmpPredicate_Case,
    left: Operand[Attribute],
    right: Operand[Attribute],
    res: Result[Attribute]
) extends DerivedOperation["db.compare", DB_CmpOp]:

  override def custom_print(printer: Printer)(using indentLevel: Int) =
    printer.print(name)
    printer.print(s" ${predicate} ")
    printer.printListF(
      operands,
      printer.printArgument
    )
    printer.print(" : ")
    printer.print(res.typ)

// ==-----== //
//   MulOp   //
// ==-----== //

object DB_MulOp:
  def name: String = "db.mul"

  // ==--- Custom Parsing ---== //
  private def inferRetType(opLeft: Attribute, opRight: Attribute) =
    (opLeft.getClass == opRight.getClass) match
      case true =>
        opLeft match
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
      case false =>
        throw new Exception("Operand types for the MulOp must be the same.")

  def parse[$: P](
      parser: Parser,
      resNames: Seq[String]
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
        resultsNames = resNames,
        resultsTypes = inferRetType(leftType, rightType),
        operandsNames = Seq(left, right),
        operandsTypes = Seq(leftType, rightType),
        attributes = z
      )
  )

  // ==----------------------== //

case class DB_MulOp(
    left: Operand[Attribute],
    right: Operand[Attribute],
    result: Result[Attribute]
) extends DerivedOperation["db.mul", DB_MulOp]:

  // added code for custom printing

  override def custom_print(printer: Printer)(using indentLevel: Int) =
    printer.print(name, " ")
    printer.printListF(operands, printer.printArgument)
    printer.print(" : ")
    printer.print(result.typ)

// ==-----== //
//   DivOp   //
// ==-----== //

object DB_DivOp:
  def name: String = "db.div"

  // ==--- Custom Parsing ---== //
  private def inferRetType(opLeft: Attribute, opRight: Attribute) =
    (opLeft.getClass == opRight.getClass) match
      case true =>
        opLeft match
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
      case false =>
        throw new Exception("Operand types for the MulOp must be the same.")

  def parse[$: P](
      parser: Parser,
      resNames: Seq[String]
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
        resultsNames = resNames,
        resultsTypes = inferRetType(leftType, rightType),
        attributes = z
      )
  )

  // ==----------------------== //

case class DB_DivOp(
    left: Operand[Attribute],
    right: Operand[Attribute],
    result: Result[Attribute]
) extends DerivedOperation["db.div", DB_DivOp]:

  override def custom_print(printer: Printer)(using indentLevel: Int) =
    printer.print(s"$name ")
    printer.printListF(
      operands,
      printer.printArgument
    )
    printer.print(" : ")
    printer.print(result.typ)

// ==-----== //
//   AddOp   //
// ==-----== //

object DB_AddOp:
  def name: String = "db.add"

  // ==--- Custom Parsing ---== //
  def parse[$: P](
      parser: Parser,
      resNames: Seq[String]
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
        resultsNames = resNames,
        resultsTypes = Seq(leftType),
        operandsNames = Seq(left, right),
        operandsTypes = Seq(leftType, rightType),
        attributes = z
      )
  )

  // ==----------------------== //

case class DB_AddOp(
    left: Operand[Attribute],
    right: Operand[Attribute],
    result: Result[Attribute]
) extends DerivedOperation["db.add", DB_AddOp]:

//added code for custom printing

  override def custom_print(printer: Printer)(using indentLevel: Int) =
    printer.print(s"$name ")
    printer.printListF(
      operands,
      printer.printArgument
    )
    printer.print(" : ")
    printer.print(result.typ)

// ==-----== //
//   SubOp   //
// ==-----== //

object DB_SubOp:
  def name: String = "db.sub"

  // ==--- Custom Parsing ---== //
  def parse[$: P](
      parser: Parser,
      resNames: Seq[String]
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
        resultsNames = resNames,
        resultsTypes = Seq(leftType),
        operandsNames = Seq(left, right),
        operandsTypes = Seq(leftType, rightType),
        attributes = z
      )
  )

  // ==----------------------== //

case class DB_SubOp(
    left: Operand[Attribute],
    right: Operand[Attribute],
    result: Result[Attribute]
) extends DerivedOperation["db.sub", DB_SubOp]:

  override def custom_print(printer: Printer)(using indentLevel: Int) =
    printer.print(s"$name ")
    printer.printListF(
      operands,
      printer.printArgument
    )
    printer.print(" : ")
    printer.print(result.typ)

// ==-----== //
//   CastOp   //
// ==-----== //

object CastOp:
  def name: String = "db.cast"

  // ==--- Custom Parsing ---== //
  def parse[$: P](
      parser: Parser,
      resNames: Seq[String]
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
        resultsNames = resNames,
        resultsTypes = resTypes,
        operandsNames = Seq(operand),
        operandsTypes = Seq(opType),
        attributes = z
      )
  )

  // ==----------------------== //

case class CastOp(
    _val: Operand[Attribute],
    res: Result[Attribute]
) extends DerivedOperation["db.cast", CastOp]:

  override def custom_print(printer: Printer)(using indentLevel: Int) =
    printer.print(s"$name ")
    printer.printArgument(_val)
    printer.print(" -> ")
    printer.print(res.typ)

val DBOps: Dialect = summonDialect[
  EmptyTuple,
  (DB_ConstantOp, DB_CmpOp, DB_MulOp, DB_DivOp, DB_SubOp, DB_AddOp, CastOp)
](Seq(DB_CharType, DB_DateType, DB_DecimalType, DB_StringType))

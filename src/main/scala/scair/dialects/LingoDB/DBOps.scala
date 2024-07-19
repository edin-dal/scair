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

object DB_DateUnit_DAY extends DB_DateUnit_Case("day")
object DB_DateUnit_MS extends DB_DateUnit_Case("millisecond")

object DB_DateUnitAttr
    extends I64EnumAttr(
      "DateUnitAttr",
      Seq(DB_DateUnit_DAY, DB_DateUnit_MS)
    )

object DB_IntervalUnit_MONTH extends DB_IntervalUnit_Case("months")
object DB_IntervalUnit_DAYTIME extends DB_IntervalUnit_Case("daytime")

object DB_IntervalUnitAttr
    extends I64EnumAttr(
      "IntervalUnitAttr",
      Seq(DB_IntervalUnit_MONTH, DB_IntervalUnit_DAYTIME)
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

object DB_DateType extends DialectAttribute {
  override def name: String = "db.date"
  override def factory = DB_CharType.apply
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

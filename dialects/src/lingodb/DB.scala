package scair.dialects.lingodb

import scair.Printer
import scair.clair.*
import scair.dialects.builtin.*
import scair.enums.*
import scair.ir.*

// ██████╗░ ██████╗░
// ██╔══██╗ ██╔══██╗
// ██║░░██║ ██████╦╝
// ██║░░██║ ██╔══██╗
// ██████╔╝ ██████╦╝
// ╚═════╝░ ╚═════╝░

/*≡==--==≡≡≡≡==--=≡≡*\
||      TYPES       ||
\*≡==---==≡≡==---==≡*/

final case class DBStringType()
    extends ParametrizedAttribute
    with TypeAttribute:
  override val name: String = "db.string"
  override val parameters: Seq[Attribute | Seq[Attribute]] = Seq()
  override def customPrint(p: Printer): Unit = p.print("!db.string")

final case class NullableType(inner: Attribute)
    extends ParametrizedAttribute
    with TypeAttribute:
  override val name: String = "db.nullable"
  override val parameters: Seq[Attribute | Seq[Attribute]] = Seq(inner)

  override def customPrint(p: Printer): Unit =
    p.print("!db.nullable<")
    p.print(inner)
    p.print(">")

final case class DecimalType(prec: IntData, scale: IntData)
    extends ParametrizedAttribute
    with TypeAttribute:
  override val name: String = "db.decimal"
  override val parameters: Seq[Attribute | Seq[Attribute]] = Seq(prec, scale)

  override def customPrint(p: Printer): Unit =
    p.print("!db.decimal<")
    p.print(prec)
    p.print(", ")
    p.print(scale)
    p.print(">")

final case class DateType(unit: StringData)
    extends ParametrizedAttribute
    with TypeAttribute:
  override val name: String = "db.date"
  override val parameters: Seq[Attribute | Seq[Attribute]] = Seq(unit)

  override def customPrint(p: Printer): Unit =
    p.print("!db.date<")
    p.print(unit.data)
    p.print(">")

final case class CharType(len: IntData)
    extends ParametrizedAttribute
    with TypeAttribute:
  override val name: String = "db.char"
  override val parameters: Seq[Attribute | Seq[Attribute]] = Seq(len)

  override def customPrint(p: Printer): Unit =
    p.print("!db.char<")
    p.print(len)
    p.print(">")

/*≡==--==≡≡≡≡==--=≡≡*\
||      ENUMS       ||
\*≡==---==≡≡==---==≡*/

enum CmpPredicate(name: String) extends I64Enum(name):
  case eq extends CmpPredicate("eq")
  case neq extends CmpPredicate("neq")
  case lt extends CmpPredicate("lt")
  case lte extends CmpPredicate("lte")
  case gt extends CmpPredicate("gt")
  case gte extends CmpPredicate("gte")

/*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  OPERATION DEFINITION  ||
\*≡==---==≡≡≡≡≡≡≡==---==≡*/

case class DBConstant(
    value: StringData,
    result: Result[Attribute],
) extends DerivedOperation["db.constant", DBConstant]
    derives DerivedOperationCompanion:

  override def customPrint(p: Printer): Unit =
    p.print("db.constant(")
    p.print(value)
    p.print(") : ")
    p.print(result.typ)

case class DBCompare(
    predicate: CmpPredicate,
    lhs: Operand[Attribute],
    rhs: Operand[Attribute],
    result: Result[Attribute],
) extends DerivedOperation["db.compare", DBCompare]
    derives DerivedOperationCompanion:

  override def customPrint(p: Printer): Unit =
    p.print("db.compare ")
    p.print(predicate.name)
    p.print(" ")
    p.print(lhs)
    p.print(":")
    p.print(lhs.typ)
    p.print(", ")
    p.print(rhs)
    p.print(":")
    p.print(rhs.typ)

case class DBAnd(
    vals: Seq[Operand[Attribute]],
    result: Result[Attribute],
) extends DerivedOperation["db.and", DBAnd] derives DerivedOperationCompanion:

  override def customPrint(p: Printer): Unit =
    p.print("db.and ")
    p.printList(vals, sep = ",")
    p.print(":")
    p.printListF(vals, v => p.print(v.typ), sep = ",")

case class DBOr(
    vals: Seq[Operand[Attribute]],
    result: Result[Attribute],
) extends DerivedOperation["db.or", DBOr] derives DerivedOperationCompanion:

  override def customPrint(p: Printer): Unit =
    p.print("db.or ")
    p.printList(vals, sep = ",")
    p.print(":")
    p.printListF(vals, v => p.print(v.typ), sep = ",")

case class DBNot(
    val_ : Operand[Attribute],
    result: Result[Attribute],
) extends DerivedOperation["db.not", DBNot] derives DerivedOperationCompanion:

  override def customPrint(p: Printer): Unit =
    p.print("db.not ")
    p.print(val_)
    p.print(" : ")
    p.print(val_.typ)

case class DBAdd(
    lhs: Operand[Attribute],
    rhs: Operand[Attribute],
    result: Result[Attribute],
) extends DerivedOperation["db.add", DBAdd] derives DerivedOperationCompanion:

  override def customPrint(p: Printer): Unit =
    p.print("db.add ")
    p.print(lhs)
    p.print(" : ")
    p.print(lhs.typ)
    p.print(", ")
    p.print(rhs)
    p.print(" : ")
    p.print(rhs.typ)

case class DBSub(
    lhs: Operand[Attribute],
    rhs: Operand[Attribute],
    result: Result[Attribute],
) extends DerivedOperation["db.sub", DBSub] derives DerivedOperationCompanion:

  override def customPrint(p: Printer): Unit =
    p.print("db.sub ")
    p.print(lhs)
    p.print(" : ")
    p.print(lhs.typ)
    p.print(", ")
    p.print(rhs)
    p.print(" : ")
    p.print(rhs.typ)

case class DBMul(
    lhs: Operand[Attribute],
    rhs: Operand[Attribute],
    result: Result[Attribute],
) extends DerivedOperation["db.mul", DBMul] derives DerivedOperationCompanion:

  override def customPrint(p: Printer): Unit =
    p.print("db.mul ")
    p.print(lhs)
    p.print(" : ")
    p.print(lhs.typ)
    p.print(", ")
    p.print(rhs)
    p.print(" : ")
    p.print(rhs.typ)

case class DBDiv(
    lhs: Operand[Attribute],
    rhs: Operand[Attribute],
    result: Result[Attribute],
) extends DerivedOperation["db.div", DBDiv] derives DerivedOperationCompanion:

  override def customPrint(p: Printer): Unit =
    p.print("db.div ")
    p.print(lhs)
    p.print(" : ")
    p.print(lhs.typ)
    p.print(", ")
    p.print(rhs)
    p.print(" : ")
    p.print(rhs.typ)

case class DBCast(
    val_ : Operand[Attribute],
    result: Result[Attribute],
) extends DerivedOperation["db.cast", DBCast] derives DerivedOperationCompanion:

  override def customPrint(p: Printer): Unit =
    p.print("db.cast ")
    p.print(val_)
    p.print(" : ")
    p.print(val_.typ)
    p.print(" -> ")
    p.print(result.typ)

val DbDialect =
  summonDialect[
    EmptyTuple,
    (
        DBConstant,
        DBCompare,
        DBAnd,
        DBOr,
        DBNot,
        DBAdd,
        DBSub,
        DBMul,
        DBDiv,
        DBCast,
    ),
  ]

package scair.dialects.lingodb

import scair.clair.*
import scair.dialects.builtin.*
import scair.enums.*
import scair.ir.*
import scair.print.Printer

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
    extends DerivedAttribute["db.string"]
    with TypeAttribute derives AttrDefs

final case class NullableType(inner: Attribute)
    extends DerivedAttribute["db.nullable"]
    with TypeAttribute derives AttrDefs

final case class DecimalType(prec: IntData, scale: IntData)
    extends DerivedAttribute["db.decimal"]
    with TypeAttribute derives AttrDefs

final case class DateType(unit: StringData)
    extends DerivedAttribute["db.date"]
    with TypeAttribute derives AttrDefs

final case class CharType(len: IntData)
    extends DerivedAttribute["db.char"]
    with TypeAttribute derives AttrDefs

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
) extends DerivedOperation["db.constant"] derives OpDefs:

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
) extends DerivedOperation["db.compare"] derives OpDefs:

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
    vals: Seq[Operand[Attribute]] = Seq.empty,
    result: Result[Attribute],
) extends DerivedOperation["db.and"] derives OpDefs:

  override def customPrint(p: Printer): Unit =
    p.print("db.and ")
    p.printList(vals, sep = ",")
    p.print(":")
    p.printListF(vals, v => p.print(v.typ), sep = ",")

case class DBOr(
    vals: Seq[Operand[Attribute]] = Seq.empty,
    result: Result[Attribute],
) extends DerivedOperation["db.or"] derives OpDefs:

  override def customPrint(p: Printer): Unit =
    p.print("db.or ")
    p.printList(vals, sep = ",")
    p.print(":")
    p.printListF(vals, v => p.print(v.typ), sep = ",")

case class DBNot(
    val_ : Operand[Attribute],
    result: Result[Attribute],
) extends DerivedOperation["db.not"] derives OpDefs:

  override def customPrint(p: Printer): Unit =
    p.print("db.not ")
    p.print(val_)
    p.print(" : ")
    p.print(val_.typ)

case class DBAdd(
    lhs: Operand[Attribute],
    rhs: Operand[Attribute],
    result: Result[Attribute],
) extends DerivedOperation["db.add"] derives OpDefs:

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
) extends DerivedOperation["db.sub"] derives OpDefs:

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
) extends DerivedOperation["db.mul"] derives OpDefs:

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
) extends DerivedOperation["db.div"] derives OpDefs:

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
) extends DerivedOperation["db.cast"] derives OpDefs:

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

package scair.dialects.lingodb

import scair.clair.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.print.Printer

// ████████╗ ██╗░░░██╗ ██████╗░ ██╗░░░░░ ███████╗ ░██████╗
// ╚══██╔══╝ ██║░░░██║ ██╔══██╗ ██║░░░░░ ██╔════╝ ██╔════╝
// ░░░██║░░░ ██║░░░██║ ██████╔╝ ██║░░░░░ █████╗░░ ╚█████╗░
// ░░░██║░░░ ██║░░░██║ ██╔═══╝░ ██║░░░░░ ██╔══╝░░ ░╚═══██╗
// ░░░██║░░░ ╚██████╔╝ ██║░░░░░ ███████╗ ███████╗ ██████╔╝
// ░░░╚═╝░░░ ░╚═════╝░ ╚═╝░░░░░ ╚══════╝ ╚══════╝ ╚═════╝░

/*≡==--==≡≡≡≡==--=≡≡*\
||      TYPES       ||
\*≡==---==≡≡==---==≡*/

final case class TupleStreamType()
    extends DerivedAttribute["tuples.tuplestream"]
    with TypeAttribute derives AttrDefs

final case class TupleType()
    extends DerivedAttribute["tuples.tuple"]
    with TypeAttribute derives AttrDefs

/*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
||      ATTRIBUTES        ||
\*≡==---==≡≡≡≡≡≡≡==---==≡*/

final case class ColumnRefAttr(scope: StringData, colName: StringData)
    extends DerivedAttribute["tuples.column_ref"] derives AttrDefs:

  override def customPrint(p: Printer): Unit =
    p.print("@")
    p.print(scope.data)
    p.print("::")
    p.print("@")
    p.print(colName.data)

final case class ColumnDefAttr(
    scope: StringData,
    colName: StringData,
    colType: Attribute,
) extends DerivedAttribute["tuples.column_def"] derives AttrDefs:

  override def customPrint(p: Printer): Unit =
    p.print("@")
    p.print(scope.data)
    p.print("::")
    p.print("@")
    p.print(colName.data)
    p.print("({type = ")
    p.print(colType)
    p.print("})")

/*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  OPERATION DEFINITION  ||
\*≡==---==≡≡≡≡≡≡≡==---==≡*/

case class GetCol(
    tuple: Operand[TupleType],
    attr: ColumnRefAttr,
    result: Result[Attribute],
) extends DerivedOperation["tuples.getcol"] derives OpDefs:

  override def customPrint(p: Printer): Unit =
    p.print("tuples.getcol ")
    p.print(tuple)
    p.print(" ")
    p.print(attr)
    p.print(" : ")
    p.print(result.typ)

case class TuplesReturn(
    results_ : Seq[Operand[Attribute]]
) extends DerivedOperation["tuples.return"]
    with IsTerminator derives OpDefs:

  override def customPrint(p: Printer): Unit =
    p.print("tuples.return ")
    p.printList(results_)
    if results_.nonEmpty then
      p.print(" : ")
      p.printListF(results_, v => p.print(v.typ))

val TuplesDialect =
  summonDialect[EmptyTuple, (GetCol, TuplesReturn)]

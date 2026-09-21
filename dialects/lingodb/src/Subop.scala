package scair.dialects.lingodb

import scair.clair.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.print.Printer

// ░██████╗ ██╗░░░██╗ ██████╗░ ░█████╗░ ██████╗░
// ██╔════╝ ██║░░░██║ ██╔══██╗ ██╔══██╗ ██╔══██╗
// ╚█████╗░ ██║░░░██║ ██████╔╝ ██║░░██║ ██████╔╝
// ░╚═══██╗ ██║░░░██║ ██╔══██╗ ██║░░██║ ██╔═══╝░
// ██████╔╝ ╚██████╔╝ ██████╔╝ ╚█████╔╝ ██║░░░░░
// ╚═════╝░ ░╚═════╝░ ╚═════╝░ ░╚════╝░ ╚═╝░░░░░

/*≡==--==≡≡≡≡==--=≡≡*\
||      TYPES       ||
\*≡==---==≡≡==---==≡*/

/** Column entry for SubopLocalTableType: prints as `name$0 : type`. */
final case class LocalTableColumn(colName: StringData, colType: Attribute)
    extends DerivedAttribute["subop.local_table_column"] derives AttrDefs:

  override def customPrint(p: Printer): Unit =
    p.print(colName.data)
    p.print("$0 : ")
    p.print(colType)

/** Prints as `!subop.local_table<[col$0 : type, ...], ["name", ...]>`. */
final case class SubopLocalTableType(
    columns: ArrayAttribute[LocalTableColumn],
    outputNames: ArrayAttribute[StringData],
) extends DerivedAttribute["subop.local_table"]
    with TypeAttribute derives AttrDefs

/*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  OPERATION DEFINITION  ||
\*≡==---==≡≡≡≡≡≡≡==---==≡*/

/** `subop.set_result 0 %val : type` */
case class SetResult(
    index: IntData,
    rel: Operand[Attribute],
) extends DerivedOperation["subop.set_result"] derives OpDefs:

  override def customPrint(p: Printer): Unit =
    p.print("subop.set_result ")
    p.print(index.value.toString)
    p.print(" ")
    p.print(rel)
    p.print(" : ")
    p.print(rel.typ)

val SubopDialect =
  summonDialect[EmptyTuple, SetResult *: EmptyTuple]

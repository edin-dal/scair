package scair.dialects.lingodb

import scair.Printer
import scair.clair.*
import scair.dialects.builtin.*
import scair.ir.*

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
    extends ParametrizedAttribute:
  override val name: String = "subop.local_table_column"

  override val parameters: Seq[Attribute | Seq[Attribute]] =
    Seq(colName, colType)

  override def customPrint(p: Printer): Unit =
    p.print(colName.data)
    p.print("$0 : ")
    p.print(colType)

/** Prints as `!subop.local_table<[col$0 : type, ...], ["name", ...]>`. */
final case class SubopLocalTableType(
    columns: Seq[LocalTableColumn],
    outputNames: Seq[StringData],
) extends ParametrizedAttribute
    with TypeAttribute:
  override val name: String = "subop.local_table"

  override val parameters: Seq[Attribute | Seq[Attribute]] =
    Seq(columns, outputNames)

  override def customPrint(p: Printer): Unit =
    p.print("!subop.local_table<[")
    p.printListF(columns, c => p.print(c))
    p.print("], [")
    p.printListF(outputNames, n => p.print(n))
    p.print("]>")

/*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  OPERATION DEFINITION  ||
\*≡==---==≡≡≡≡≡≡≡==---==≡*/

/** `subop.set_result 0 %val : type` */
case class SetResult(
    index: IntData,
    rel: Operand[Attribute],
) extends DerivedOperation["subop.set_result", SetResult]
    derives DerivedOperationCompanion:

  override def customPrint(p: Printer): Unit =
    p.print("subop.set_result ")
    p.print(index.value.toString)
    p.print(" ")
    p.print(rel)
    p.print(" : ")
    p.print(rel.typ)

val SubopDialect =
  summonDialect[EmptyTuple, SetResult *: EmptyTuple]

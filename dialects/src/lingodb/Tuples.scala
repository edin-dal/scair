package scair.dialects.lingodb

import scair.Printer
import scair.clair.*
import scair.dialects.builtin.*
import scair.ir.*

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
    extends ParametrizedAttribute
    with TypeAttribute:
  override val name: String = "tuples.tuplestream"
  override val parameters: Seq[Attribute | Seq[Attribute]] = Seq()
  override def customPrint(p: Printer): Unit = p.print("!tuples.tuplestream")

final case class TupleType() extends ParametrizedAttribute with TypeAttribute:
  override val name: String = "tuples.tuple"
  override val parameters: Seq[Attribute | Seq[Attribute]] = Seq()
  override def customPrint(p: Printer): Unit = p.print("!tuples.tuple")

/*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
||      ATTRIBUTES        ||
\*≡==---==≡≡≡≡≡≡≡==---==≡*/

final case class ColumnRefAttr(scope: StringData, colName: StringData)
    extends ParametrizedAttribute:
  override val name: String = "tuples.column_ref"

  override val parameters: Seq[Attribute | Seq[Attribute]] =
    Seq(scope, colName)

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
) extends ParametrizedAttribute:
  override val name: String = "tuples.column_def"

  override val parameters: Seq[Attribute | Seq[Attribute]] =
    Seq(scope, colName, colType)

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

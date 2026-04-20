package scair.dialects.lingodb

import scair.clair.*
import scair.dialects.builtin.*
import scair.enums.*
import scair.ir.*
import scair.print.Printer

// ██████╗░ ███████╗ ██╗░░░░░ ░█████╗░ ██╗░░░░░ ░██████╗░
// ██╔══██╗ ██╔════╝ ██║░░░░░ ██╔══██╗ ██║░░░░░ ██╔════╝░
// ██████╔╝ █████╗░░ ██║░░░░░ ███████║ ██║░░░░░ ██║░░██╗░
// ██╔══██╗ ██╔══╝░░ ██║░░░░░ ██╔══██║ ██║░░░░░ ██║░░╚██╗
// ██║░░██║ ███████╗ ███████╗ ██║░░██║ ███████╗ ╚██████╔╝
// ╚═╝░░╚═╝ ╚══════╝ ╚══════╝ ╚═╝░░╚═╝ ╚══════╝ ░╚═════╝░

/*≡==--==≡≡≡≡==--=≡≡*\
||      ENUMS       ||
\*≡==---==≡≡==---==≡*/

enum AggrFunc(name: String) extends I64Enum(name):
  case sum extends AggrFunc("sum")
  case min extends AggrFunc("min")
  case max extends AggrFunc("max")
  case avg extends AggrFunc("avg")
  case count extends AggrFunc("count")

enum SetSemantic(name: String) extends I64Enum(name):
  case distinct extends SetSemantic("distinct")
  case all extends SetSemantic("all")

enum SortSpec(name: String) extends I64Enum(name):
  case asc extends SortSpec("asc")
  case desc extends SortSpec("desc")

/*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
||      ATTRIBUTES        ||
\*≡==---==≡≡≡≡≡≡≡==---==≡*/

final case class SortSpecificationAttr(colRef: ColumnRefAttr, spec: SortSpec)
    extends ParametrizedAttribute:
  override val name: String = "relalg.sort_spec"

  override val parameters: Seq[Attribute | Seq[Attribute]] =
    Seq(colRef, spec)

  override def customPrint(p: Printer): Unit =
    p.print("(")
    p.print(colRef)
    p.print(",")
    p.print(spec.name)
    p.print(")")

/*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
||       HELPERS          ||
\*≡==---==≡≡≡≡≡≡≡==---==≡*/

/** Print a region's body with block arguments shown inline before the brace,
  * matching LingoDB's custom format: `(%arg0: !tuples.tuple) {\n  ops...\n}`
  * instead of generic MLIR: `{\n^bb0(%0: !tuples.tuple):\n  ops...\n}`
  */
private def printRegionWithInlineArgs(
    lp: Printer,
    region: Region,
): Unit =
  region.blocks.toList match
    case entry :: others =>
      lp.print("(")
      lp.printListF(entry.arguments, lp.printArgument)
      lp.print(") {\n")
      entry.operations.foreach(x => lp.indented(lp.print(x)))
      others.foreach(lp.print)
      lp.withIndent(lp.print("}"))
    case _ =>
      lp.print("{\n")
      lp.withIndent(lp.print("}"))

/*≡==--==≡≡≡≡≡≡≡≡≡==--=≡≡*\
||  OPERATION DEFINITION  ||
\*≡==---==≡≡≡≡≡≡≡==---==≡*/

case class BaseTable(
    tableId: StringData,
    columns: DictionaryAttr,
    result: Result[TupleStreamType],
) extends DerivedOperation["relalg.basetable"] derives OpDefs:

  override def customPrint(p: Printer): Unit =
    p.print("relalg.basetable {table_identifier = ")
    p.print(tableId)
    p.print("} columns: {")
    p.printListF(
      columns.entries,
      (k, v) =>
        p.print(k); p.print(" => "); p.print(v)
      ,
      sep = ", ",
    )
    p.print("}")

case class Selection(
    rel: Operand[TupleStreamType],
    predicate: Region,
    result: Result[TupleStreamType],
) extends DerivedOperation["relalg.selection"] derives OpDefs:

  override def customPrint(p: Printer): Unit =
    val lp = p.scoped
    lp.print("relalg.selection ")
    lp.print(rel)
    lp.print(" ")
    printRegionWithInlineArgs(lp, predicate)

case class MapOp(
    rel: Operand[TupleStreamType],
    computedCols: ArrayAttribute[ColumnDefAttr],
    lambda: Region,
    result: Result[TupleStreamType],
) extends DerivedOperation["relalg.map"] derives OpDefs:

  override def customPrint(p: Printer): Unit =
    val lp = p.scoped
    lp.print("relalg.map ")
    lp.print(rel)
    lp.print(" computes : [")
    lp.printListF(computedCols.attrValues, c => lp.print(c))
    lp.print("] ")
    printRegionWithInlineArgs(lp, lambda)

case class InnerJoin(
    left: Operand[TupleStreamType],
    right: Operand[TupleStreamType],
    predicate: Region,
    result: Result[TupleStreamType],
) extends DerivedOperation["relalg.join"] derives OpDefs:

  override def customPrint(p: Printer): Unit =
    val lp = p.scoped
    lp.print("relalg.join ")
    lp.print(left)
    lp.print(", ")
    lp.print(right)
    lp.print("  ")
    printRegionWithInlineArgs(lp, predicate)

case class CrossProduct(
    left: Operand[TupleStreamType],
    right: Operand[TupleStreamType],
    result: Result[TupleStreamType],
) extends DerivedOperation["relalg.crossproduct"] derives OpDefs:

  override def customPrint(p: Printer): Unit =
    p.print("relalg.crossproduct ")
    p.print(left)
    p.print(", ")
    p.print(right)

case class Aggregation(
    rel: Operand[TupleStreamType],
    groupByCols: ArrayAttribute[ColumnRefAttr],
    computedCols: ArrayAttribute[ColumnDefAttr],
    aggr: Region,
    result: Result[TupleStreamType],
) extends DerivedOperation["relalg.aggregation"] derives OpDefs:

  override def customPrint(p: Printer): Unit =
    val lp = p.scoped
    lp.print("relalg.aggregation ")
    lp.print(rel)
    lp.print(" [")
    lp.printListF(groupByCols.attrValues, c => lp.print(c))
    lp.print("] computes : [")
    lp.printListF(computedCols.attrValues, c => lp.print(c))
    lp.print("] ")
    printRegionWithInlineArgs(lp, aggr)

case class AggrFn(
    rel: Operand[TupleStreamType],
    fn: AggrFunc,
    attr: ColumnRefAttr,
    result: Result[Attribute],
) extends DerivedOperation["relalg.aggrfn"] derives OpDefs:

  override def customPrint(p: Printer): Unit =
    p.print("relalg.aggrfn ")
    p.print(fn.name)
    p.print(" ")
    p.print(attr)
    p.print(" ")
    p.print(rel)
    p.print(" : ")
    p.print(result.typ)

case class CountRows(
    rel: Operand[TupleStreamType],
    result: Result[Attribute],
) extends DerivedOperation["relalg.count"] derives OpDefs:

  override def customPrint(p: Printer): Unit =
    p.print("relalg.count ")
    p.print(rel)

case class Sort(
    rel: Operand[TupleStreamType],
    sortspecs: ArrayAttribute[SortSpecificationAttr],
    result: Result[TupleStreamType],
) extends DerivedOperation["relalg.sort"] derives OpDefs:

  override def customPrint(p: Printer): Unit =
    p.print("relalg.sort ")
    p.print(rel)
    p.print(" [")
    p.printListF(sortspecs.attrValues, s => p.print(s), sep = ",")
    p.print("]")

case class Limit(
    rel: Operand[TupleStreamType],
    maxRows: IntData,
    result: Result[TupleStreamType],
) extends DerivedOperation["relalg.limit"] derives OpDefs:

  override def customPrint(p: Printer): Unit =
    p.print("relalg.limit ")
    p.print(maxRows.value.toString)
    p.print(" ")
    p.print(rel)

case class Projection(
    rel: Operand[TupleStreamType],
    setSemantic: SetSemantic,
    cols: ArrayAttribute[ColumnRefAttr],
    result: Result[TupleStreamType],
) extends DerivedOperation["relalg.projection"] derives OpDefs:

  override def customPrint(p: Printer): Unit =
    p.print("relalg.projection ")
    p.print(setSemantic.name)
    p.print(" [")
    p.printListF(cols.attrValues, c => p.print(c))
    p.print("] ")
    p.print(rel)

case class Materialize(
    rel: Operand[TupleStreamType],
    cols: ArrayAttribute[ColumnRefAttr],
    colNames: ArrayAttribute[StringData],
    result: Result[Attribute],
) extends DerivedOperation["relalg.materialize"] derives OpDefs:

  override def customPrint(p: Printer): Unit =
    p.print("relalg.materialize ")
    p.print(rel)
    p.print(" [")
    p.printListF(cols.attrValues, c => p.print(c))
    p.print("] => [")
    p.printListF(colNames.attrValues, n => p.print(n))
    p.print("] : ")
    p.print(result.typ)

/** `%0 = relalg.query (){ ... relalg.query_return %N : type } -> type` */
case class RelAlgQuery(
    body: Region,
    result: Result[Attribute],
) extends DerivedOperation["relalg.query"] derives OpDefs:

  override def customPrint(p: Printer): Unit =
    val lp = p.scoped
    lp.print("relalg.query (){\n")
    body.blocks.toList match
      case entry :: others =>
        entry.operations.foreach(x => lp.indented(lp.print(x)))
        others.foreach(lp.print)
      case _ => ()
    lp.withIndent(lp.print("} -> "))
    lp.print(result.typ)

/** `relalg.query_return %val : type` — terminator for RelAlgQuery regions. */
case class QueryReturn(
    rel: Operand[Attribute]
) extends DerivedOperation["relalg.query_return"]
    with IsTerminator derives OpDefs:

  override def customPrint(p: Printer): Unit =
    p.print("relalg.query_return ")
    p.print(rel)
    p.print(" : ")
    p.print(rel.typ)

val RelAlgDialect =
  summonDialect[
    EmptyTuple,
    (
        BaseTable,
        Selection,
        MapOp,
        InnerJoin,
        CrossProduct,
        Aggregation,
        AggrFn,
        CountRows,
        Sort,
        Limit,
        Projection,
        Materialize,
        RelAlgQuery,
        QueryReturn,
    ),
  ]

package scair.dialects.LingoDB.RelAlgOps

import fastparse.*
import scair.AttrParser
import scair.AttrParser.whitespace
import scair.EnumAttr.I64EnumAttr
import scair.EnumAttr.I64EnumAttrCase
import scair.Parser
import scair.Parser.BareId
import scair.Parser.E
import scair.Parser.ValueId
import scair.Parser.mapTry
import scair.Parser.orElse
import scair.Printer
import scair.clair.macros.DerivedOperation
import scair.clair.macros.summonDialect
import scair.dialects.LingoDB.TupleStream.*
import scair.dialects.builtin.*
import scair.ir.*

// ==---== //
//  Enums
// ==---== //

abstract class RelAlg_SortSpec_Case(override val symbol: String)
    extends I64EnumAttrCase(symbol)

abstract class RelAlg_GroupJoinBehaviour_Case(override val symbol: String)
    extends I64EnumAttrCase(symbol)

abstract class RelAlg_AggrFunc_Case(override val symbol: String)
    extends I64EnumAttrCase(symbol)

abstract class RelAlg_SetSemantic_Case(override val symbol: String)
    extends I64EnumAttrCase(symbol)

object RelAlg_SortSpec_DESC extends RelAlg_SortSpec_Case("desc")
object RelAlg_SortSpec_ASC extends RelAlg_SortSpec_Case("asc")

object RelAlg_SortSpec
    extends I64EnumAttr(
      "SortSpec",
      Seq(RelAlg_SortSpec_DESC, RelAlg_SortSpec_ASC)
    )

object RelAlg_GroupJoinBehaviour_Inner
    extends RelAlg_GroupJoinBehaviour_Case("inner")

object RelAlg_GroupJoinBehaviour_Outer
    extends RelAlg_GroupJoinBehaviour_Case("outer")

object RelAlg_GroupJoinBehaviour
    extends I64EnumAttr(
      "GroupJoinBehaviour",
      Seq(RelAlg_GroupJoinBehaviour_Inner, RelAlg_GroupJoinBehaviour_Outer)
    )

object RelAlg_Func_Sum extends RelAlg_AggrFunc_Case("sum")
object RelAlg_Func_Min extends RelAlg_AggrFunc_Case("min")
object RelAlg_Func_Max extends RelAlg_AggrFunc_Case("max")
object RelAlg_Func_Avg extends RelAlg_AggrFunc_Case("avg")
object RelAlg_Func_Count extends RelAlg_AggrFunc_Case("count")
object RelAlg_Func_Any extends RelAlg_AggrFunc_Case("any")
object RelAlg_Func_StdDevSamp extends RelAlg_AggrFunc_Case("stddev_samp")
object RelAlg_Func_VarSamp extends RelAlg_AggrFunc_Case("var_samp")

object RelAlg_AggrFunc
    extends I64EnumAttr(
      "AggrFunc",
      Seq(
        RelAlg_Func_Sum,
        RelAlg_Func_Min,
        RelAlg_Func_Max,
        RelAlg_Func_Avg,
        RelAlg_Func_Count,
        RelAlg_Func_Any,
        RelAlg_Func_StdDevSamp,
        RelAlg_Func_VarSamp
      )
    )

object RelAlg_SetSemantic_Distinct extends RelAlg_SetSemantic_Case("distinct")
object RelAlg_SetSemantic_All extends RelAlg_SetSemantic_Case("all")

object RelAlg_SetSemantic
    extends I64EnumAttr(
      "SortSpec",
      Seq(RelAlg_SetSemantic_Distinct, RelAlg_SetSemantic_All)
    )

////////////////
// ATTRIBUTES //
////////////////

// ==---------------------== //
//   SortSpecificationAttr   //
// ==---------------------== //

object SortSpecificationAttr extends AttributeCompanion:
  override def name: String = "db.sortspec"

  override def parse[$: P](parser: AttrParser): P[Attribute] =
    P(
      "(" ~ ColumnRefAttr.parse(parser) ~ "," ~ RelAlg_SortSpec.caseParser ~ ")"
    )
      .map((x, y) =>
        SortSpecificationAttr(
          x.asInstanceOf[ColumnRefAttr],
          y.asInstanceOf[RelAlg_SortSpec_Case]
        )
      )

case class SortSpecificationAttr(
    val attr: ColumnRefAttr,
    val sortSpec: RelAlg_SortSpec_Case
) extends ParametrizedAttribute:
  override def name: String = "db.interval"
  override def parameters: Seq[Attribute | Seq[Attribute]] = Seq(attr, sortSpec)

  override def custom_print(p: Printer) =
    p.print("(", attr, ",", sortSpec, ")")(using indentLevel = 0)

///////////
// TYPES //
///////////

////////////////
// OPERATIONS //
////////////////

// ==------------------== //
//   Private Region def   //
// ==------------------== //

private def DialectRegion[$: P](parser: Parser) = P(
  E({ parser.enterLocalRegion })
    ~ (parser.BlockArgList
      .orElse(Seq())
      .mapTry((x: Seq[(String, Attribute)]) =>

        val b = new Block()
        b.arguments ++= x map parser.currentScope.defineBlockArgument
        b
      )
      ~ "{"
      ~ parser.Operations(1) ~ "}").map((b: Block, y: Seq[Operation]) =>
      b.operations ++= y
      Region(b)
    )
)
  ~ E({ parser.enterParentRegion })

// ==-----------== //
//   BaseTableOp   //
// ==-----------== //

object BaseTableOp:
  def name: String = "relalg.basetable"

  // ==--- Custom Parsing ---== //
  def parse[$: P](
      parser: Parser,
      resNames: Seq[String]
  ) = P(
    parser.OptionalAttributes ~
      "columns" ~ ":" ~ "{" ~ (BareId ~ "=>" ~ ColumnDefAttr.parse(parser))
        .rep(0, sep = ",")
        .map(Map(_*)) ~ "}"
  ).map(
    (
        x: Map[String, Attribute],
        y: Map[String, Attribute]
    ) =>
      parser.generateOperation(
        opName = name,
        resultsNames = resNames,
        resultsTypes = Seq(TupleStreamType(Seq())),
        attributes = x,
        properties = y
      )
  )

  // ==----------------------== //

case class BaseTableOp(
    table_identifier: StringData,
    columns: DictionaryAttr,
    result: Result[TupleStreamType]
) extends DerivedOperation["relalg.basetable", BaseTableOp]

// ==-----------== //
//   SelectionOp   //
// ==-----------== //

object SelectionOp:
  def name: String = "relalg.selection"

  // ==--- Custom Parsing ---== //
  def parse[$: P](
      parser: Parser,
      resNames: Seq[String]
  ): P[Operation] = P(
    ValueId ~ DialectRegion(parser) ~
      parser.OptionalKeywordAttributes
  )
    .map(
      (
          x: String,
          y: Region,
          z: Map[String, Attribute]
      ) =>
        val operand_type = parser.currentScope.valueMap(x).typ
        parser.generateOperation(
          opName = name,
          operandsNames = Seq(x),
          operandsTypes = Seq(operand_type),
          resultsNames = resNames,
          resultsTypes = Seq(TupleStreamType(Seq())),
          regions = Seq(y),
          attributes = z
        )
    )

  // ==----------------------== //

case class SelectionOp(
    rel: Operand[TupleStreamType],
    predicate: Region,
    result: Result[TupleStreamType]
) extends DerivedOperation["relalg.selection", SelectionOp]

// ==-----== //
//   MapOp   //
// ==-----== //

object MapOp:
  def name: String = "relalg.map"

  // ==--- Custom Parsing ---== //
  def parse[$: P](
      parser: Parser,
      resNames: Seq[String]
  ): P[Operation] = P(
    ValueId
      ~ "computes" ~ ":"
      ~ "[" ~ ColumnDefAttr.parse(parser).rep.map(ArrayAttribute(_)) ~ "]"
      ~ DialectRegion(parser)
      ~ parser.OptionalKeywordAttributes
  ).map(
    (
        x: String,
        z: Attribute,
        y: Region,
        w: Map[String, Attribute]
    ) =>
      val operand_type = parser.currentScope.valueMap(x).typ
      parser.generateOperation(
        opName = name,
        operandsNames = Seq(x),
        operandsTypes = Seq(operand_type),
        resultsNames = resNames,
        resultsTypes = Seq(TupleStreamType(Seq())),
        regions = Seq(y),
        attributes = w + ("computed_cols" -> z)
      )
  )

  // ==----------------------== //

case class MapOp(
    computed_cols: ArrayAttribute[Attribute],
    rel: Operand[TupleStreamType],
    predicate: Region,
    result: Result[TupleStreamType]
) extends DerivedOperation["relalg.map", MapOp]

// ==-------------== //
//   AggregationOp   //
// ==-------------== //

object AggregationOp:
  def name: String = "relalg.aggregation"

  // ==--- Custom Parsing ---== //
  def parse[$: P](
      parser: Parser,
      resNames: Seq[String]
  ): P[Operation] = P(
    ValueId
      ~ "[" ~ ColumnRefAttr
        .parse(parser)
        .rep(sep = ",")
        .map(ArrayAttribute(_)) ~ "]"
      ~ "computes" ~ ":"
      ~ "[" ~ ColumnDefAttr
        .parse(parser)
        .rep(sep = ",")
        .map(ArrayAttribute(_)) ~ "]"
      ~ DialectRegion(parser)
      ~ parser.OptionalKeywordAttributes
  ).map(
    (
        x: String,
        reff: Attribute,
        deff: Attribute,
        y: Region,
        w: Map[String, Attribute]
    ) =>
      val operand_type = parser.currentScope.valueMap(x).typ
      parser.generateOperation(
        opName = name,
        operandsNames = Seq(x),
        operandsTypes = Seq(operand_type),
        resultsNames = resNames,
        resultsTypes = Seq(TupleStreamType(Seq())),
        regions = Seq(y),
        attributes = w + ("group_by_cols" -> reff) + ("computed_cols" -> deff)
      )
  )

  // ==----------------------== //

case class AggregationOp(
    group_by_cols: ArrayAttribute[Attribute],
    computed_cols: ArrayAttribute[Attribute],
    rel: Operand[TupleStreamType],
    aggr_func: Region,
    result: Result[TupleStreamType]
) extends DerivedOperation["relalg.aggregation", AggregationOp]

// ==-----------== //
//   CountRowsOp   //
// ==-----------== //

object CountRowsOp:
  def name: String = "relalg.count"

  // ==--- Custom Parsing ---== //
  def parse[$: P](
      parser: Parser,
      resNames: Seq[String]
  ): P[Operation] = P(
    ValueId ~ parser.OptionalAttributes
  ).map(
    (
        x: String,
        y: Map[String, Attribute]
    ) =>
      val operand_type = parser.currentScope.valueMap(x).typ
      parser.generateOperation(
        opName = name,
        operandsNames = Seq(x),
        operandsTypes = Seq(operand_type),
        resultsNames = resNames,
        resultsTypes = Seq(I64),
        attributes = y
      )
  )

  // ==----------------------== //

case class CountRowsOp(
    rel: Operand[TupleStreamType],
    res: Result[IntegerType]
) extends DerivedOperation["relalg.count", CountRowsOp]

// ==----------== //
//   AggrFuncOp   //
// ==----------== //

object AggrFuncOp:
  def name: String = "relalg.aggrfn"

  // ==--- Custom Parsing ---== //
  def parse[$: P](
      parser: Parser,
      resNames: Seq[String]
  ): P[Operation] = P(
    RelAlg_AggrFunc.caseParser ~ ColumnRefAttr.parse(parser)
      ~ ValueId ~ ":" ~ parser.Type.rep(1)
      ~ parser.OptionalAttributes
  ).map(
    (
        aggrfunc: Attribute,
        attr: Attribute,
        x: String,
        resTypes: Seq[Attribute],
        y: Map[String, Attribute]
    ) =>
      val operand_type = parser.currentScope.valueMap(x).typ
      parser.generateOperation(
        opName = name,
        operandsNames = Seq(x),
        operandsTypes = Seq(operand_type),
        resultsNames = resNames,
        resultsTypes = resTypes,
        attributes = y + ("fn" -> aggrfunc) + ("attr" -> attr)
      )
  )

  // ==----------------------== //

case class AggrFuncOp(
    fn: RelAlg_AggrFunc_Case,
    attr: ColumnRefAttr,
    rel: Operand[TupleStreamType],
    result: Result[Attribute]
) extends DerivedOperation["relalg.aggrfn", AggrFuncOp]

// ==------== //
//   SortOp   //
// ==------== //

object SortOp:
  def name: String = "relalg.sort"

  // ==--- Custom Parsing ---== //
  def parse[$: P](
      parser: Parser,
      resNames: Seq[String]
  ): P[Operation] = P(
    ValueId
      ~ "[" ~ (SortSpecificationAttr
        .parse(parser))
        .rep(sep = ",")
        .map(ArrayAttribute(_)) ~ "]"
      ~ parser.OptionalAttributes
  ).map(
    (
        x: String,
        attr: Attribute,
        y: Map[String, Attribute]
    ) =>
      val operand_type = parser.currentScope.valueMap(x).typ
      parser.generateOperation(
        opName = name,
        operandsNames = Seq(x),
        operandsTypes = Seq(operand_type),
        resultsNames = resNames,
        resultsTypes = Seq(TupleStreamType(Seq())),
        attributes = y + ("sortspecs" -> attr)
      )
  )

  // ==----------------------== //

case class SortOp(
    sortspecs: ArrayAttribute[Attribute],
    rel: Operand[TupleStreamType],
    result: Result[TupleStreamType]
) extends DerivedOperation["relalg.sort", SortOp]

// ==-------------== //
//   MaterializeOp   //
// ==-------------== //

object MaterializeOp:
  def name: String = "relalg.materialize"

  // ==--- Custom Parsing ---== //
  def parse[$: P](
      parser: Parser,
      resNames: Seq[String]
  ): P[Operation] = P(
    ValueId
      ~ "[" ~ ColumnRefAttr
        .parse(parser)
        .rep(sep = ",")
        .map(ArrayAttribute(_)) ~ "]"
      ~ "=" ~ ">"
      ~ parser.ArrayAttributeP
      ~ ":"
      ~ parser.Type.rep
      ~ parser.OptionalAttributes
  ).map(
    (
        operand: String,
        cols: Attribute,
        columns: Attribute,
        resTypes: Seq[Attribute],
        y: Map[String, Attribute]
    ) =>
      val operand_type = parser.currentScope.valueMap(operand).typ
      parser.generateOperation(
        opName = name,
        operandsNames = Seq(operand),
        operandsTypes = Seq(operand_type),
        resultsNames = resNames,
        resultsTypes = resTypes,
        attributes = y + ("cols" -> cols) + ("columns" -> columns)
      )
  )

  // ==----------------------== //

case class MaterializeOp(
    cols: ArrayAttribute[Attribute],
    columns: ArrayAttribute[Attribute],
    rel: Operand[TupleStreamType],
    result: Result[Attribute]
) extends DerivedOperation["relalg.materialize", MaterializeOp]

val RelAlgOps: Dialect =
  summonDialect[
    EmptyTuple,
    (
        BaseTableOp,
        SelectionOp,
        MapOp,
        AggregationOp,
        CountRowsOp,
        AggrFuncOp,
        SortOp,
        MaterializeOp
    )
  ]()

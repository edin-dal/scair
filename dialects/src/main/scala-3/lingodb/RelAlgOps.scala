package scair.dialects.LingoDB.RelAlgOps

import fastparse.*
import scair.AttrParser
import scair.EnumAttr.I64EnumAttr
import scair.EnumAttr.I64EnumAttrCase
import scair.Parser
import scair.Parser.BareId
import scair.Parser.E
import scair.Parser.ValueId
import scair.Parser.mapTry
import scair.Parser.orElse
import scair.Parser.whitespace
import scair.dialects.LingoDB.SubOperatorOps.*
import scair.dialects.LingoDB.TupleStream.*
import scair.dialects.builtin.*
import scair.exceptions.VerifyException
import scair.ir.*

import scala.collection.immutable
import scala.collection.mutable

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

object SortSpecificationAttr extends AttributeObject {
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

}

case class SortSpecificationAttr(
    val attr: ColumnRefAttr,
    val sortSpec: RelAlg_SortSpec_Case
) extends ParametrizedAttribute(
      name = "db.interval"
    ) {
  override def custom_print = s"(${attr.custom_print},${sortSpec.custom_print})"
}

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
      .mapTry((x: Seq[(String, Attribute)]) => {
        val b = new Block(Seq.from(x.map(_._2)), Seq.empty)
        parser.currentScope.defineValues(x.map(_._1) zip b.arguments)
        b
      })
      ~ "{"
      ~ parser.Operations(1) ~ "}").map((b: Block, y: Seq[Operation]) => {
      b.operations ++= y
      new Region(Seq(b))
    })
)
  ~ E({ parser.enterParentRegion })

// ==-----------== //
//   BaseTableOp   //
// ==-----------== //

object BaseTableOp extends OperationCompanion {
  override def name: String = "relalg.basetable"

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      parser: Parser
  ) = P(
    parser.OptionalAttributes ~
      "columns" ~ ":" ~ "{" ~ (BareId ~ "=>" ~ ColumnDefAttr.parse(parser))
        .rep(0, sep = ",")
        .map(DictType(_*)) ~ "}"
  ).map(
    (
        x: DictType[String, Attribute],
        y: DictType[String, Attribute]
    ) =>
      parser.generateOperation(
        opName = name,
        resultsTypes = Seq(TupleStream(Seq())),
        attributes = x,
        properties = y
      )
  )
  // ==----------------------== //

}

case class BaseTableOp(
    override val operands: Seq[Value[Attribute]],
    override val successors: Seq[Block],
    override val results_types: Seq[Attribute],
    override val regions: Seq[Region],
    override val properties: DictType[String, Attribute],
    override val attributes: DictType[String, Attribute]
) extends BaseOperation(
      name = "relalg.basetable",
      operands,
      successors,
      results_types,
      regions,
      properties,
      attributes
    ) {

  override def custom_verify(): Unit = (
    operands.length,
    successors.length,
    results.length,
    regions.length
  ) match {
    case (0, 0, 1, 0) =>
      results(0).typ match {
        case _: TupleStream =>
        case _ =>
          throw new VerifyException(
            "BaseTableOp Operation must contain only 1 result."
          )
      }
      attributes.get("table_identifier") match {
        case Some(x) =>
          x match {
            case _: StringData =>
            case _ =>
              throw new VerifyException(
                "BaseTableOp Operation must contain a StringAttr named 'table_identifier'."
              )
          }
        case None =>
          throw new VerifyException(
            "BaseTableOp Operation must contain a StringAttr named 'table_identifier'."
          )
      }
  }

}

// ==-----------== //
//   SelectionOp   //
// ==-----------== //

object SelectionOp extends OperationCompanion {
  override def name: String = "relalg.selection"

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      parser: Parser
  ): P[Operation] = P(
    ValueId ~ DialectRegion(parser) ~
      parser.OptionalKeywordAttributes
  )
    .map(
      (
          x: String,
          y: Region,
          z: DictType[String, Attribute]
      ) =>
        val operand_type = parser.currentScope.valueMap(x).typ
        parser.generateOperation(
          opName = name,
          operandsNames = Seq(x),
          operandsTypes = Seq(operand_type),
          resultsTypes = Seq(TupleStream(Seq())),
          regions = Seq(y),
          attributes = z
        )
    )
  // ==----------------------== //

}

case class SelectionOp(
    override val operands: Seq[Value[Attribute]],
    override val successors: Seq[Block],
    override val results_types: Seq[Attribute],
    override val regions: Seq[Region],
    override val properties: DictType[String, Attribute],
    override val attributes: DictType[String, Attribute]
) extends BaseOperation(
      name = "relalg.selection",
      operands,
      successors,
      results_types,
      regions,
      properties,
      attributes
    ) {

  override def custom_verify(): Unit = (
    operands.length,
    successors.length,
    results.length,
    regions.length
  ) match {
    case (1, 0, 1, 1) =>
      operands(0).typ match {
        case _: TupleStream =>
        case _ =>
          throw new VerifyException(
            "SelectionOp Operation must contain only 1 operand of type TupleStream."
          )
      }
      results(0).typ match {
        case _: TupleStream =>
        case _ =>
          throw new VerifyException(
            "SelectionOp Operation must contain only 1 result of type TupleStream."
          )
      }
  }

}

// ==-----== //
//   MapOp   //
// ==-----== //

object MapOp extends OperationCompanion {
  override def name: String = "relalg.map"

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      parser: Parser
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
        w: DictType[String, Attribute]
    ) =>
      val operand_type = parser.currentScope.valueMap(x).typ
      parser.generateOperation(
        opName = name,
        operandsNames = Seq(x),
        operandsTypes = Seq(operand_type),
        resultsTypes = Seq(TupleStream(Seq())),
        regions = Seq(y),
        attributes = w + ("computed_cols" -> z)
      )
  )
  // ==----------------------== //

}

case class MapOp(
    override val operands: Seq[Value[Attribute]],
    override val successors: Seq[Block],
    override val results_types: Seq[Attribute],
    override val regions: Seq[Region],
    override val properties: DictType[String, Attribute],
    override val attributes: DictType[String, Attribute]
) extends BaseOperation(
      name = "relalg.map",
      operands,
      successors,
      results_types,
      regions,
      properties,
      attributes
    ) {

  override def custom_verify(): Unit = (
    operands.length,
    successors.length,
    results.length,
    regions.length
  ) match {
    case (1, 0, 1, 1) =>
      operands(0).typ match {
        case _: TupleStream =>
        case _ =>
          throw new VerifyException(
            "MapOp Operation must contain only 1 operand of type TupleStream."
          )
      }
      results(0).typ match {
        case _: TupleStream =>
        case _ =>
          throw new VerifyException(
            "MapOp Operation must contain only 1 result of type TupleStream."
          )
      }

      attributes.get("computed_cols") match {
        case Some(x) =>
          x match {
            case _: ArrayAttribute[_] =>
            case _ =>
              throw new VerifyException(
                "MapOp Operation must contain a ArrayAttribute named 'computed_cols'."
              )
          }
        case None =>
          throw new VerifyException(
            "MapOp Operation must contain a ArrayAttribute named 'computed_cols'."
          )
      }
  }

}

// ==-------------== //
//   AggregationOp   //
// ==-------------== //

object AggregationOp extends OperationCompanion {
  override def name: String = "relalg.aggregation"

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      parser: Parser
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
        w: DictType[String, Attribute]
    ) =>
      val operand_type = parser.currentScope.valueMap(x).typ
      parser.generateOperation(
        opName = name,
        operandsNames = Seq(x),
        operandsTypes = Seq(operand_type),
        resultsTypes = Seq(TupleStream(Seq())),
        regions = Seq(y),
        attributes = w + ("group_by_cols" -> reff, "computed_cols" -> deff)
      )
  )
  // ==----------------------== //

}

case class AggregationOp(
    override val operands: Seq[Value[Attribute]],
    override val successors: Seq[Block],
    override val results_types: Seq[Attribute],
    override val regions: Seq[Region],
    override val properties: DictType[String, Attribute],
    override val attributes: DictType[String, Attribute]
) extends BaseOperation(
      name = "relalg.aggregation",
      operands,
      successors,
      results_types,
      regions,
      properties,
      attributes
    ) {

  override def custom_verify(): Unit = (
    operands.length,
    successors.length,
    results.length,
    regions.length
  ) match {
    case (1, 0, 1, 1) =>
      operands(0).typ match {
        case _: TupleStream =>
        case _ =>
          throw new VerifyException(
            "AggregationOp Operation must contain only 1 operand of type TupleStream."
          )
      }
      results(0).typ match {
        case _: TupleStream =>
        case _ =>
          throw new VerifyException(
            "AggregationOp Operation must contain only 1 result of type TupleStream."
          )
      }

      attributes.get("computed_cols") match {
        case Some(x) =>
          x match {
            case _: ArrayAttribute[_] =>
            case _ =>
              throw new VerifyException(
                "AggregationOp Operation must contain an ArrayAttribute named 'computed_cols'."
              )
          }
        case _ =>
          throw new VerifyException(
            "AggregationOp Operation must contain an ArrayAttribute named 'computed_cols'."
          )
      }

      attributes.get("group_by_cols") match {
        case Some(x) =>
          x match {
            case _: ArrayAttribute[_] =>
            case _ =>
              throw new VerifyException(
                "AggregationOp Operation must contain an ArrayAttribute named 'group_by_cols'."
              )
          }
        case _ =>
          throw new VerifyException(
            "AggregationOp Operation must contain an ArrayAttribute named 'group_by_cols'."
          )
      }
  }

}

// ==-----------== //
//   CountRowsOp   //
// ==-----------== //

object CountRowsOp extends OperationCompanion {
  override def name: String = "relalg.count"

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      parser: Parser
  ): P[Operation] = P(
    ValueId ~ parser.OptionalAttributes
  ).map(
    (
        x: String,
        y: DictType[String, Attribute]
    ) =>
      val operand_type = parser.currentScope.valueMap(x).typ
      parser.generateOperation(
        opName = name,
        operandsNames = Seq(x),
        operandsTypes = Seq(operand_type),
        resultsTypes = Seq(I64),
        attributes = y
      )
  )
  // ==----------------------== //

}

case class CountRowsOp(
    override val operands: Seq[Value[Attribute]],
    override val successors: Seq[Block],
    override val results_types: Seq[Attribute],
    override val regions: Seq[Region],
    override val properties: DictType[String, Attribute],
    override val attributes: DictType[String, Attribute]
) extends BaseOperation(
      name = "relalg.count",
      operands,
      successors,
      results_types,
      regions,
      properties,
      attributes
    ) {

  override def custom_verify(): Unit = (
    operands.length,
    successors.length,
    results.length,
    regions.length
  ) match {
    case (1, 0, 1, 0) =>
      operands(0).typ match {
        case _: TupleStream =>
        case _ =>
          throw new VerifyException(
            "CountRowsOp Operation must contain 1 operand of type TupleStream."
          )
      }
      results(0).typ match {
        case _: IntegerType =>
        case _ =>
          throw new VerifyException(
            "CountRowsOp Operation must contain only 1 result of IntegerType."
          )
      }
    case _ =>
      throw new VerifyException(
        "CountRowsOp Operation must contain only 1 operand and 1 result."
      )
  }

}

// ==----------== //
//   AggrFuncOp   //
// ==----------== //

object AggrFuncOp extends OperationCompanion {
  override def name: String = "relalg.aggrfn"

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      parser: Parser
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
        y: DictType[String, Attribute]
    ) =>
      val operand_type = parser.currentScope.valueMap(x).typ
      parser.generateOperation(
        opName = name,
        operandsNames = Seq(x),
        operandsTypes = Seq(operand_type),
        resultsTypes = resTypes,
        attributes = y + ("fn" -> aggrfunc, "attr" -> attr)
      )
  )
  // ==----------------------== //

}

case class AggrFuncOp(
    override val operands: Seq[Value[Attribute]],
    override val successors: Seq[Block],
    override val results_types: Seq[Attribute],
    override val regions: Seq[Region],
    override val properties: DictType[String, Attribute],
    override val attributes: DictType[String, Attribute]
) extends BaseOperation(
      name = "relalg.aggrfn",
      operands,
      successors,
      results_types,
      regions,
      properties,
      attributes
    ) {

  override def custom_verify(): Unit = (
    operands.length,
    successors.length,
    results.length,
    regions.length
  ) match {
    case (1, 0, 1, 0) =>
      operands(0).typ match {
        case _: TupleStream =>
        case _ =>
          throw new VerifyException(
            "AggrFuncOp Operation must contain 1 operand of type TupleStream."
          )
      }
      attributes.get("fn") match {
        case Some(x) =>
          x match {
            case _: RelAlg_AggrFunc_Case =>
            case _ =>
              throw new VerifyException(
                "AggrFuncOp Operation must contain an RelAlg_AggrFunc enum named 'fn'."
              )
          }
        case _ =>
          throw new VerifyException(
            "AggrFuncOp Operation must contain an RelAlg_AggrFunc enum named 'fn'."
          )
      }

      attributes.get("attr") match {
        case Some(x) =>
          x match {
            case _: ColumnRefAttr =>
            case _ =>
              throw new VerifyException(
                "AggrFuncOp Operation must contain an ColumnRefAttr named 'attr'."
              )
          }
        case _ =>
          throw new VerifyException(
            "AggrFuncOp Operation must contain an ColumnRefAttr named 'attr'."
          )
      }
    case _ =>
      throw new VerifyException(
        "AggrFuncOp Operation must contain only 1 operand and 1 result."
      )
  }

}

// ==------== //
//   SortOp   //
// ==------== //

object SortOp extends OperationCompanion {
  override def name: String = "relalg.sort"

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      parser: Parser
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
        y: DictType[String, Attribute]
    ) =>
      val operand_type = parser.currentScope.valueMap(x).typ
      parser.generateOperation(
        opName = name,
        operandsNames = Seq(x),
        operandsTypes = Seq(operand_type),
        resultsTypes = Seq(TupleStream(Seq())),
        attributes = y + ("sortspecs" -> attr)
      )
  )
  // ==----------------------== //

}

case class SortOp(
    override val operands: Seq[Value[Attribute]],
    override val successors: Seq[Block],
    override val results_types: Seq[Attribute],
    override val regions: Seq[Region],
    override val properties: DictType[String, Attribute],
    override val attributes: DictType[String, Attribute]
) extends BaseOperation(
      name = "relalg.sort",
      operands,
      successors,
      results_types,
      regions,
      properties,
      attributes
    ) {

  override def custom_verify(): Unit = (
    operands.length,
    successors.length,
    results.length,
    regions.length
  ) match {
    case (1, 0, 1, 0) =>
      operands(0).typ match {
        case _: TupleStream =>
        case _ =>
          throw new VerifyException(
            "SortOp Operation must contain 1 operand of type TupleStream."
          )
      }
      results(0).typ match {
        case _: TupleStream =>
        case _ =>
          throw new VerifyException(
            "SortOp Operation must contain 1 operand of type TupleStream."
          )
      }
      attributes.get("sortspecs") match {
        case Some(x) =>
          x match {
            case _: ArrayAttribute[_] =>
            case _ =>
              throw new VerifyException(
                "SortOp Operation must contain an ArrayAttribute enum named 'sortspecs'."
              )
          }
        case _ =>
          throw new VerifyException(
            "SortOp Operation must contain an ArrayAttribute enum named 'sortspecs'."
          )
      }
    case _ =>
      throw new VerifyException(
        "SortOp Operation must contain only 1 operand and 1 result."
      )
  }

}

// ==-------------== //
//   MaterializeOp   //
// ==-------------== //

object MaterializeOp extends OperationCompanion {
  override def name: String = "relalg.materialize"

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      parser: Parser
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
        y: DictType[String, Attribute]
    ) =>
      val operand_type = parser.currentScope.valueMap(operand).typ
      parser.generateOperation(
        opName = name,
        operandsNames = Seq(operand),
        operandsTypes = Seq(operand_type),
        resultsTypes = resTypes,
        attributes = y + ("cols" -> cols, "columns" -> columns)
      )
  )
  // ==----------------------== //

}

case class MaterializeOp(
    override val operands: Seq[Value[Attribute]],
    override val successors: Seq[Block],
    override val results_types: Seq[Attribute],
    override val regions: Seq[Region],
    override val properties: DictType[String, Attribute],
    override val attributes: DictType[String, Attribute]
) extends BaseOperation(
      name = "relalg.materialize",
      operands,
      successors,
      results_types,
      regions,
      properties,
      attributes
    ) {

  override def custom_verify(): Unit = (
    operands.length,
    successors.length,
    results.length,
    regions.length
  ) match {
    case (1, 0, 1, 0) =>
      operands(0).typ match {
        case _: TupleStream =>
        case _ =>
          throw new VerifyException(
            "MaterializeOp Operation must contain 1 operand of type TupleStream."
          )
      }
      results(0).typ match {
        case _: ResultTable =>
        case _ =>
          throw new VerifyException(
            "MaterializeOp Operation must contain 1 operand of type ResultOp."
          )
      }

      attributes.get("cols") match {
        case Some(x) =>
          x match {
            case _: ArrayAttribute[_] =>
            case _ =>
              throw new VerifyException(
                "MaterializeOp Operation must contain an ArrayAttribute enum named 'cols'."
              )
          }
        case _ =>
          throw new VerifyException(
            "MaterializeOp Operation must contain an ArrayAttribute enum named 'cols'."
          )
      }

      attributes.get("columns") match {
        case Some(x) =>
          x match {
            case _: ArrayAttribute[_] =>
            case _ =>
              throw new VerifyException(
                "MaterializeOp Operation must contain an ArrayAttribute named 'columns'."
              )
          }
        case _ =>
          throw new VerifyException(
            "MaterializeOp Operation must contain an ArrayAttribute named 'columns'."
          )
      }
    case _ =>
      throw new VerifyException(
        "MaterializeOp Operation must contain only 1 operand and 1 result."
      )
  }

}

val RelAlgOps: Dialect =
  new Dialect(
    operations = Seq(
      BaseTableOp,
      SelectionOp,
      MapOp,
      AggregationOp,
      CountRowsOp,
      AggrFuncOp,
      SortOp,
      MaterializeOp
    ),
    attributes = Seq()
  )

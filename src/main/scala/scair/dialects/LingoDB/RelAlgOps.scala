package scair.dialects.LingoDB.RelAlgOps

import scair.dialects.LingoDB.TupleStream._
import scair.dialects.LingoDB.SubOperatorOps._
import scair.dialects.LingoDB.DBOps._

import fastparse._
import scair.EnumAttr.{I64EnumAttrCase, I64EnumAttr}
import scair.dialects.builtin._
import scala.collection.immutable
import scala.collection.mutable
import scair.dialects.irdl.{Operand, OpResult}
import scair.Parser.{
  whitespace,
  E,
  BlockArgList,
  optionlessSeq,
  giveBack,
  Scope,
  ValueId,
  BareId,
  Type,
  DictionaryAttribute,
  AttributeEntry
}
import scair.AttrParser.{ArrayAttributeP}
import scair.{
  ListType,
  DictType,
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

object SortSpecificationAttr extends DialectAttribute {
  override def name: String = "db.sortspec"
  override def parse[$: P]: P[Attribute] =
    P("(" ~ ColumnRefAttr.parse ~ "," ~ RelAlg_SortSpec.caseParser ~ ")")
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
  override def toString = s"(${attr},${sortSpec})"
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
    ~ BlockArgList.?.map(optionlessSeq)
      .map(parser.defineBlockValues) ~ "{"
    ~ parser.OperationPat.rep(1) ~ "}"
    ~ E({ parser.enterParentRegion })
).map((x: ListType[Value[Attribute]], y: Seq[Operation]) =>
  new Region(Seq(new Block(y, x)))
)

// ==-----------== //
//   BaseTableOp   //
// ==-----------== //

object BaseTableOp extends DialectOperation {
  override def name: String = "relalg.basetable"
  override def factory = BaseTableOp.apply

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      resNames: Seq[String],
      parser: Parser
  ) = P(
    DictionaryAttribute.?.map(Parser.optionlessSeq) ~
      "columns" ~ ":" ~ "{" ~ (BareId ~ "=>" ~ ColumnDefAttr.parse)
        .rep(0, sep = ",") ~ "}"
  ).map(
    (
        x: Seq[(String, Attribute)],
        y: Seq[(String, Attribute)]
    ) =>
      parser.verifyCustomOp(
        opGen = factory,
        opName = name,
        resultNames = resNames,
        resultTypes = (for { name <- resNames } yield TupleStream(Seq())),
        dictAttrs = x,
        dictProps = y
      )
  )
  // ==----------------------== //
}

case class BaseTableOp(
    override val operands: ListType[Value[Attribute]],
    override val successors: ListType[Block],
    override val results: ListType[Value[Attribute]],
    override val regions: ListType[Region],
    override val dictionaryProperties: DictType[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute]
) extends RegisteredOperation(name = "relalg.basetable") {

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
          throw new Exception(
            "BaseTableOp Operation must contain only 1 result."
          )
      }
      dictionaryAttributes.get("table_identifier") match {
        case Some(x) =>
          x match {
            case _: StringAttribute =>
            case _ =>
              throw new Exception(
                "BaseTableOp Operation must contain a StringAttr named 'table_identifier'."
              )
          }
        case None =>
          throw new Exception(
            "BaseTableOp Operation must contain a StringAttr named 'table_identifier'."
          )
      }
  }
}

// ==-----------== //
//   SelectionOp   //
// ==-----------== //

object SelectionOp extends DialectOperation {
  override def name: String = "relalg.selection"
  override def factory = SelectionOp.apply

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      resNames: Seq[String],
      parser: Parser
  ): P[Operation] = P(
    ValueId ~ DialectRegion(parser)
      ~ ("attributes" ~ DictionaryAttribute).?.map(
        optionlessSeq
      )
  ).map(
    (
        x: String,
        y: Region,
        z: Seq[(String, Attribute)]
    ) =>
      parser.verifyCustomOp(
        opGen = factory,
        opName = name,
        operandNames = Seq(x),
        resultNames = resNames,
        resultTypes = (for { name <- resNames } yield TupleStream(Seq())),
        regions = Seq(y),
        dictAttrs = z,
        noForwardOperandRef = 1
      )
  )
  // ==----------------------== //
}

case class SelectionOp(
    override val operands: ListType[Value[Attribute]],
    override val successors: ListType[Block],
    override val results: ListType[Value[Attribute]],
    override val regions: ListType[Region],
    override val dictionaryProperties: DictType[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute]
) extends RegisteredOperation(name = "relalg.selection") {

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
          throw new Exception(
            "SelectionOp Operation must contain only 1 operand of type TupleStream."
          )
      }
      results(0).typ match {
        case _: TupleStream =>
        case _ =>
          throw new Exception(
            "SelectionOp Operation must contain only 1 result of type TupleStream."
          )
      }
  }
}

// ==-----== //
//   MapOp   //
// ==-----== //

object MapOp extends DialectOperation {
  override def name: String = "relalg.map"
  override def factory = MapOp.apply

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      resNames: Seq[String],
      parser: Parser
  ): P[Operation] = P(
    ValueId
      ~ "computes" ~ ":"
      ~ "[" ~ ColumnDefAttr.parse.rep.map(ArrayAttribute(_)) ~ "]"
      ~ DialectRegion(parser)
      ~ ("attributes" ~ DictionaryAttribute).?.map(optionlessSeq)
  ).map(
    (
        x: String,
        z: Attribute,
        y: Region,
        w: Seq[(String, Attribute)]
    ) =>
      parser.verifyCustomOp(
        opGen = factory,
        opName = name,
        operandNames = Seq(x),
        resultNames = resNames,
        resultTypes = (for { name <- resNames } yield TupleStream(Seq())),
        regions = Seq(y),
        dictAttrs = w :+ ("computed_cols", z),
        noForwardOperandRef = 1
      )
  )
  // ==----------------------== //
}

case class MapOp(
    override val operands: ListType[Value[Attribute]],
    override val successors: ListType[Block],
    override val results: ListType[Value[Attribute]],
    override val regions: ListType[Region],
    override val dictionaryProperties: DictType[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute]
) extends RegisteredOperation(name = "relalg.map") {

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
          throw new Exception(
            "MapOp Operation must contain only 1 operand of type TupleStream."
          )
      }
      results(0).typ match {
        case _: TupleStream =>
        case _ =>
          throw new Exception(
            "MapOp Operation must contain only 1 result of type TupleStream."
          )
      }

      dictionaryAttributes.get("computed_cols") match {
        case Some(x) =>
          x match {
            case _: ArrayAttribute[_] =>
            case _ =>
              throw new Exception(
                "MapOp Operation must contain a ArrayAttribute named 'computed_cols'."
              )
          }
        case None =>
          throw new Exception(
            "MapOp Operation must contain a ArrayAttribute named 'computed_cols'."
          )
      }
  }
}

// ==-------------== //
//   AggregationOp   //
// ==-------------== //

object AggregationOp extends DialectOperation {
  override def name: String = "relalg.aggregation"
  override def factory = AggregationOp.apply

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      resNames: Seq[String],
      parser: Parser
  ): P[Operation] = P(
    ValueId
      ~ "[" ~ ColumnRefAttr.parse.rep(sep = ",").map(ArrayAttribute(_)) ~ "]"
      ~ "computes" ~ ":"
      ~ "[" ~ ColumnDefAttr.parse.rep(sep = ",").map(ArrayAttribute(_)) ~ "]"
      ~ DialectRegion(parser)
      ~ ("attributes" ~ DictionaryAttribute).?.map(optionlessSeq)
  ).map(
    (
        x: String,
        reff: Attribute,
        deff: Attribute,
        y: Region,
        w: Seq[(String, Attribute)]
    ) =>
      parser.verifyCustomOp(
        opGen = factory,
        opName = name,
        operandNames = Seq(x),
        resultNames = resNames,
        resultTypes = (for { name <- resNames } yield TupleStream(Seq())),
        regions = Seq(y),
        dictAttrs = w :+ ("group_by_cols", reff) :+ ("computed_cols", deff),
        noForwardOperandRef = 1
      )
  )
  // ==----------------------== //
}

case class AggregationOp(
    override val operands: ListType[Value[Attribute]],
    override val successors: ListType[Block],
    override val results: ListType[Value[Attribute]],
    override val regions: ListType[Region],
    override val dictionaryProperties: DictType[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute]
) extends RegisteredOperation(name = "relalg.aggregation") {

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
          throw new Exception(
            "AggregationOp Operation must contain only 1 operand of type TupleStream."
          )
      }
      results(0).typ match {
        case _: TupleStream =>
        case _ =>
          throw new Exception(
            "AggregationOp Operation must contain only 1 result of type TupleStream."
          )
      }

      dictionaryAttributes.get("computed_cols") match {
        case Some(x) =>
          x match {
            case _: ArrayAttribute[_] =>
            case _ =>
              throw new Exception(
                "AggregationOp Operation must contain an ArrayAttribute named 'computed_cols'."
              )
          }
        case _ =>
          throw new Exception(
            "AggregationOp Operation must contain an ArrayAttribute named 'computed_cols'."
          )
      }

      dictionaryAttributes.get("group_by_cols") match {
        case Some(x) =>
          x match {
            case _: ArrayAttribute[_] =>
            case _ =>
              throw new Exception(
                "AggregationOp Operation must contain an ArrayAttribute named 'group_by_cols'."
              )
          }
        case _ =>
          throw new Exception(
            "AggregationOp Operation must contain an ArrayAttribute named 'group_by_cols'."
          )
      }
  }
}

// ==-----------== //
//   CountRowsOp   //
// ==-----------== //

object CountRowsOp extends DialectOperation {
  override def name: String = "relalg.count"
  override def factory = CountRowsOp.apply

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      resNames: Seq[String],
      parser: Parser
  ): P[Operation] = P(
    ValueId ~ DictionaryAttribute.?.map(Parser.optionlessSeq)
  ).map(
    (
        x: String,
        y: Seq[(String, Attribute)]
    ) =>
      parser.verifyCustomOp(
        opGen = factory,
        opName = name,
        operandNames = Seq(x),
        resultNames = resNames,
        resultTypes = Seq(I64),
        dictAttrs = y,
        noForwardOperandRef = 1
      )
  )
  // ==----------------------== //
}

case class CountRowsOp(
    override val operands: ListType[Value[Attribute]],
    override val successors: ListType[Block],
    override val results: ListType[Value[Attribute]],
    override val regions: ListType[Region],
    override val dictionaryProperties: DictType[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute]
) extends RegisteredOperation(name = "relalg.count") {

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
          throw new Exception(
            "CountRowsOp Operation must contain 1 operand of type TupleStream."
          )
      }
      results(0).typ match {
        case _: IntegerType =>
        case _ =>
          throw new Exception(
            "CountRowsOp Operation must contain only 1 result of IntegerType."
          )
      }
    case _ =>
      throw new Exception(
        "CountRowsOp Operation must contain only 1 operand and 1 result."
      )
  }
}

// ==----------== //
//   AggrFuncOp   //
// ==----------== //

object AggrFuncOp extends DialectOperation {
  override def name: String = "relalg.aggrfn"
  override def factory = AggrFuncOp.apply

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      resNames: Seq[String],
      parser: Parser
  ): P[Operation] = P(
    RelAlg_AggrFunc.caseParser ~ ColumnRefAttr.parse
      ~ ValueId ~ ":" ~ Type.rep(1)
      ~ DictionaryAttribute.?.map(optionlessSeq)
  ).map(
    (
        aggrfunc: Attribute,
        attr: Attribute,
        x: String,
        resTypes: Seq[Attribute],
        y: Seq[(String, Attribute)]
    ) =>
      parser.verifyCustomOp(
        opGen = factory,
        opName = name,
        operandNames = Seq(x),
        resultNames = resNames,
        resultTypes = resTypes,
        dictAttrs = y :+ ("fn", aggrfunc) :+ ("attr", attr),
        noForwardOperandRef = 1
      )
  )
  // ==----------------------== //
}

case class AggrFuncOp(
    override val operands: ListType[Value[Attribute]],
    override val successors: ListType[Block],
    override val results: ListType[Value[Attribute]],
    override val regions: ListType[Region],
    override val dictionaryProperties: DictType[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute]
) extends RegisteredOperation(name = "relalg.aggrfn") {

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
          throw new Exception(
            "AggrFuncOp Operation must contain 1 operand of type TupleStream."
          )
      }
      dictionaryAttributes.get("fn") match {
        case Some(x) =>
          x match {
            case _: RelAlg_AggrFunc_Case =>
            case _ =>
              throw new Exception(
                "AggrFuncOp Operation must contain an RelAlg_AggrFunc enum named 'fn'."
              )
          }
        case _ =>
          throw new Exception(
            "AggrFuncOp Operation must contain an RelAlg_AggrFunc enum named 'fn'."
          )
      }

      dictionaryAttributes.get("attr") match {
        case Some(x) =>
          x match {
            case _: ColumnRefAttr =>
            case _ =>
              throw new Exception(
                "AggrFuncOp Operation must contain an ColumnRefAttr named 'attr'."
              )
          }
        case _ =>
          throw new Exception(
            "AggrFuncOp Operation must contain an ColumnRefAttr named 'attr'."
          )
      }
    case _ =>
      throw new Exception(
        "AggrFuncOp Operation must contain only 1 operand and 1 result."
      )
  }
}

// ==------== //
//   SortOp   //
// ==------== //

object SortOp extends DialectOperation {
  override def name: String = "relalg.sort"
  override def factory = SortOp.apply

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      resNames: Seq[String],
      parser: Parser
  ): P[Operation] = P(
    ValueId
      ~ "[" ~ (SortSpecificationAttr.parse)
        .rep(sep = ",")
        .map(ArrayAttribute(_)) ~ "]"
      ~ DictionaryAttribute.?.map(optionlessSeq)
  ).map(
    (
        x: String,
        attr: Attribute,
        y: Seq[(String, Attribute)]
    ) =>
      parser.verifyCustomOp(
        opGen = factory,
        opName = name,
        operandNames = Seq(x),
        resultNames = resNames,
        resultTypes = (for { name <- resNames } yield TupleStream(Seq())),
        dictAttrs = y :+ ("sortspecs", attr),
        noForwardOperandRef = 1
      )
  )
  // ==----------------------== //
}

case class SortOp(
    override val operands: ListType[Value[Attribute]],
    override val successors: ListType[Block],
    override val results: ListType[Value[Attribute]],
    override val regions: ListType[Region],
    override val dictionaryProperties: DictType[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute]
) extends RegisteredOperation(name = "relalg.sort") {

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
          throw new Exception(
            "SortOp Operation must contain 1 operand of type TupleStream."
          )
      }
      results(0).typ match {
        case _: TupleStream =>
        case _ =>
          throw new Exception(
            "SortOp Operation must contain 1 operand of type TupleStream."
          )
      }
      dictionaryAttributes.get("sortspecs") match {
        case Some(x) =>
          x match {
            case _: ArrayAttribute[_] =>
            case _ =>
              throw new Exception(
                "SortOp Operation must contain an ArrayAttribute enum named 'sortspecs'."
              )
          }
        case _ =>
          throw new Exception(
            "SortOp Operation must contain an ArrayAttribute enum named 'sortspecs'."
          )
      }
    case _ =>
      throw new Exception(
        "SortOp Operation must contain only 1 operand and 1 result."
      )
  }
}

// ==-------------== //
//   MaterializeOp   //
// ==-------------== //

object MaterializeOp extends DialectOperation {
  override def name: String = "relalg.materialize"
  override def factory = MaterializeOp.apply

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      resNames: Seq[String],
      parser: Parser
  ): P[Operation] = P(
    ValueId
      ~ "[" ~ ColumnRefAttr.parse.rep(sep = ",").map(ArrayAttribute(_)) ~ "]"
      ~ "=" ~ ">"
      ~ ArrayAttributeP
      ~ ":"
      ~ Type.rep
      ~ DictionaryAttribute.?.map(optionlessSeq)
  ).map(
    (
        operand: String,
        cols: Attribute,
        columns: Attribute,
        resTypes: Seq[Attribute],
        y: Seq[(String, Attribute)]
    ) =>
      parser.verifyCustomOp(
        opGen = factory,
        opName = name,
        operandNames = Seq(operand),
        resultNames = resNames,
        resultTypes = resTypes,
        dictAttrs = y :+ ("cols", cols) :+ ("columns", columns),
        noForwardOperandRef = 1
      )
  )
  // ==----------------------== //
}

case class MaterializeOp(
    override val operands: ListType[Value[Attribute]],
    override val successors: ListType[Block],
    override val results: ListType[Value[Attribute]],
    override val regions: ListType[Region],
    override val dictionaryProperties: DictType[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute]
) extends RegisteredOperation(name = "relalg.materialize") {

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
          throw new Exception(
            "MaterializeOp Operation must contain 1 operand of type TupleStream."
          )
      }
      results(0).typ match {
        case _: ResultTable =>
        case _ =>
          throw new Exception(
            "MaterializeOp Operation must contain 1 operand of type ResultOp."
          )
      }

      dictionaryAttributes.get("cols") match {
        case Some(x) =>
          x match {
            case _: ArrayAttribute[_] =>
            case _ =>
              throw new Exception(
                "MaterializeOp Operation must contain an ArrayAttribute enum named 'cols'."
              )
          }
        case _ =>
          throw new Exception(
            "MaterializeOp Operation must contain an ArrayAttribute enum named 'cols'."
          )
      }

      dictionaryAttributes.get("columns") match {
        case Some(x) =>
          x match {
            case _: ArrayAttribute[_] =>
            case _ =>
              throw new Exception(
                "MaterializeOp Operation must contain an ArrayAttribute named 'columns'."
              )
          }
        case _ =>
          throw new Exception(
            "MaterializeOp Operation must contain an ArrayAttribute named 'columns'."
          )
      }
    case _ =>
      throw new Exception(
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

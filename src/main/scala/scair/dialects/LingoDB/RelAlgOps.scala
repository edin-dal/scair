package scair.dialects.LingoDB.RelAlgOps

import scair.dialects.LingoDB.TupleStream._
import scair.dialects.LingoDB.SubOperatorOps._
import scair.dialects.LingoDB.DBOps._

import fastparse._
import scair.EnumAttr.{I64EnumAttrCase, I64EnumAttr}
import scair.dialects.builtin._
import scala.collection.immutable
import scair.dialects.irdl.{Operand, OpResult}
import scair.Parser.{
  whitespace,
  E,
  BlockArgList,
  optionlessSeq,
  Scope,
  ValueId,
  BareId,
  Type,
  DictionaryAttribute,
  AttributeEntry
}
import scair.{
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

// ==-------------== //
//   ColumnDefAttr   //
// ==-------------== //

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
  "(" ~ E({ parser.enterLocalRegion })
    ~ BlockArgList.?.map(optionlessSeq)
      .map(parser.defineBlockValues) ~ ")" ~ "{"
    ~ parser.OperationPat.rep(1) ~ "}"
    ~ E({ parser.enterParentRegion })
).map((x: Seq[Value[Attribute]], y: Seq[Operation]) =>
  new Region(Seq(new Block(y, x)))
)

// ==-----------== //
//   BaseTableOp   //
// ==-----------== //

object BaseTableOp extends DialectOperation {
  override def name: String = "subop.basetable"
  override def factory = BaseTableOp.apply

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      resNames: Seq[String],
      parser: Parser
  ): P[Operation] = P(
    DictionaryAttribute.?.map(Parser.optionlessSeq) ~
      "columns" ~ ":" ~ "{" ~ (BareId ~ "=>" ~ ColumnDefAttr.parse).rep(0) ~ "}"
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
    override val operands: collection.mutable.ArrayBuffer[Value[Attribute]],
    override val successors: collection.mutable.ArrayBuffer[Block],
    override val results: Seq[Value[Attribute]],
    override val regions: Seq[Region],
    override val dictionaryProperties: immutable.Map[String, Attribute],
    override val dictionaryAttributes: immutable.Map[String, Attribute]
) extends RegisteredOperation(name = "subop.basetable") {

  override def verify(): Unit = (
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
  }
}

// ==-----------== //
//   SelectionOp   //
// ==-----------== //

object RelAlgOps {
  def main(args: Array[String]): Unit = {}
}

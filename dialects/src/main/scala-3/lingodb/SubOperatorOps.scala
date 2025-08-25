package scair.dialects.LingoDB.SubOperatorOps

import fastparse.*
import scair.AttrParser
import scair.Parser
import scair.Parser.BareId
import scair.Parser.ValueId
import scair.Parser.whitespace
import scair.Printer
import scair.dialects.builtin.*
import scair.ir.*

////////////////
// ATTRIBUTES //
////////////////

// ==------------== //
//   StateMembers   //
// ==------------== //

object StateMembers extends AttributeCompanion {
  override def name: String = "subop.state_members"

  override def parse[$: P](parser: AttrParser) = P(
    "[" ~ (BareId.map(StringData(_)) ~ ":" ~ parser.Type)
      .rep(0, sep = ",") ~ "]"
  ).map((x: Seq[(StringData, Attribute)]) => {
    val (names, types) = x.unzip
    StateMembers(names, types)
  })

}

case class StateMembers(
    val names: Seq[StringData],
    val types: Seq[Attribute]
) extends ParametrizedAttribute {

  override def name: String = "subop.state_members"

  override def parameters: Seq[Attribute | Seq[Attribute]] =
    names :++ types

  override def custom_print(p: Printer) =
    p.printListF(
      names zip types,
      (n, t) => p.print(n.data, " : ", t)(using indentLevel = 0),
      "[",
      ", ",
      "]"
    )

    // s"[${(for { (x, y) <- (names zip types) } yield s"${x.stringLiteral} : ${y.custom_print}").mkString(", ")}]"

}

///////////
// TYPES //
///////////

// ==-----------== //
//   ResultTable   //
// ==-----------== //

object ResultTable extends AttributeCompanion {
  override def name: String = "subop.result_table"

  override def parse[$: P](parser: AttrParser) = P(
    "<" ~ StateMembers.parse(parser) ~ ">"
  ).map(x => ResultTable(x.asInstanceOf[StateMembers]))

}

case class ResultTable(
    val members: StateMembers
) extends ParametrizedAttribute
    with TypeAttribute {

  override def name: String = "subop.result_table"
  override def parameters: Seq[Attribute | Seq[Attribute]] = Seq(members)
}

////////////////
// OPERATIONS //
////////////////

// ==-----------== //
//   SetResultOp   //
// ==-----------== //

object SetResultOp extends OperationCompanion {
  override def name: String = "subop.set_result"

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      parser: Parser
  ): P[Operation] = P(
    parser.Attribute ~ ValueId ~ ":" ~ parser.Type
      ~ parser.OptionalAttributes
  ).map(
    (
        x: Attribute,
        y: String,
        z: Attribute,
        w: Map[String, Attribute]
    ) =>
      parser.generateOperation(
        opName = name,
        operandsNames = Seq(y),
        operandsTypes = Seq(z),
        attributes = w + ("result_id" -> x)
      )
  )
  // ==----------------------== //

}

case class SetResultOp(
    override val operands: Seq[Value[Attribute]],
    override val successors: Seq[Block],
    override val results: Seq[Result[Attribute]],
    override val regions: Seq[Region],
    override val properties: Map[String, Attribute],
    override val attributes: DictType[String, Attribute]
) extends BaseOperation(
      name = "subop.set_result",
      operands,
      successors,
      results,
      regions,
      properties,
      attributes
    ) {

  override def custom_verify(): Either[String, Operation] = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    properties.size
  ) match {
    case (1, 0, 0, 0, 0) =>
      attributes.get("result_id") match {
        case Some(x) =>
          x match {
            case _: IntegerAttr => Right(this)
            case _              =>
              Left(
                "SetResultOp Operation's 'result_id' must be of type IntegerAttr."
              )
          }
        case _ =>
          Left(
            "SetResultOp Operation's 'result_id' must be of type IntegerAttr."
          )
      }
    case _ =>
      Left(
        "SetResultOp Operation must have 1 operand, 0 successors, 0 results, 0 regions and 0 properties."
      )
  }

}

val SubOperatorOps: Dialect =
  new Dialect(
    operations = Seq(SetResultOp),
    attributes = Seq(ResultTable)
  )

// !subop.result_table<[avg_disc$0 : !db.decimal<31, 21>, count_order$0 : i64]>

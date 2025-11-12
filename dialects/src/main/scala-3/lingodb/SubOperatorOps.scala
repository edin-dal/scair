package scair.dialects.LingoDB.SubOperatorOps

import fastparse.*
import scair.AttrParser
import scair.AttrParser.whitespace
import scair.Parser
import scair.Parser.BareId
import scair.Parser.ValueId
import scair.Printer
import scair.clair.macros.DerivedOperation
import scair.clair.macros.summonDialect
import scair.dialects.builtin.*
import scair.ir.*

////////////////
// ATTRIBUTES //
////////////////

// ==------------== //
//   StateMembers   //
// ==------------== //

object StateMembers extends AttributeCompanion:
  override def name: String = "subop.state_members"

  override def parse[$: P](parser: AttrParser) = P(
    "[" ~ (BareId.map(StringData(_)) ~ ":" ~ parser.Type)
      .rep(0, sep = ",") ~ "]"
  ).map((x: Seq[(StringData, Attribute)]) =>
    val (names, types) = x.unzip
    StateMembers(names, types)
  )

case class StateMembers(
    val names: Seq[StringData],
    val types: Seq[Attribute]
) extends ParametrizedAttribute:

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

///////////
// TYPES //
///////////

// ==-----------== //
//   ResultTable   //
// ==-----------== //

object ResultTable extends AttributeCompanion:
  override def name: String = "subop.result_table"

  override def parse[$: P](parser: AttrParser) = P(
    "<" ~ StateMembers.parse(parser) ~ ">"
  ).map(x => ResultTable(x.asInstanceOf[StateMembers]))

case class ResultTable(
    val members: StateMembers
) extends ParametrizedAttribute
    with TypeAttribute:

  override def name: String = "subop.result_table"
  override def parameters: Seq[Attribute | Seq[Attribute]] = Seq(members)

////////////////
// OPERATIONS //
////////////////

// ==-----------== //
//   SetResultOp   //
// ==-----------== //

object SetResultOp:
  def name: String = "subop.set_result"

  // ==--- Custom Parsing ---== //
  def parse[$: P](
      parser: Parser,
      resNames: Seq[String]
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
        attributes = w + ("result_id" -> x),
        resultsNames = resNames
      )
  )

  // ==----------------------== //

case class SetResultOp(
    result_id: IntegerAttr,
    state: Operand[Attribute]
) extends DerivedOperation["subop.set_result", SetResultOp]
// !subop.result_table<[avg_disc$0 : !db.decimal<31, 21>, count_order$0 : i64]>

val SubOperatorOps: Dialect = summonDialect[EmptyTuple, Tuple1[SetResultOp]](
  Seq(StateMembers, ResultTable)
)

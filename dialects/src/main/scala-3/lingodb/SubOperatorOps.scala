package scair.dialects.LingoDB.SubOperatorOps

import fastparse.*
import scair.AttrParser
import scair.Parser
import scair.Parser.BareId
import scair.Parser.ValueId
import scair.Parser.whitespace
import scair.dialects.builtin.*
import scair.exceptions.VerifyException
import scair.ir.*

import scala.collection.immutable
import scala.collection.mutable

////////////////
// ATTRIBUTES //
////////////////

// ==------------== //
//   StateMembers   //
// ==------------== //

object StateMembers extends AttributeObject {
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
) extends ParametrizedAttribute(
      name = "subop.state_members",
      parameters = names :++ types
    ) {

  override def custom_print =
    s"[${(for { (x, y) <- (names zip types) } yield s"${x.stringLiteral} : ${y.custom_print}").mkString(", ")}]"

}

///////////
// TYPES //
///////////

// ==-----------== //
//   ResultTable   //
// ==-----------== //

object ResultTable extends AttributeObject {
  override def name: String = "subop.result_table"

  override def parse[$: P](parser: AttrParser) = P(
    "<" ~ StateMembers.parse(parser) ~ ">"
  ).map(x => ResultTable(x.asInstanceOf[StateMembers]))

}

case class ResultTable(
    val members: StateMembers
) extends ParametrizedAttribute(
      name = "subop.result_table",
      parameters = Seq(members)
    )
    with TypeAttribute

////////////////
// OPERATIONS //
////////////////

// ==-----------== //
//   SetResultOp   //
// ==-----------== //

object SetResultOp extends MLIROperationObject {
  override def name: String = "subop.set_result"
  override def factory = SetResultOp.apply

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      parser: Parser
  ): P[MLIROperation] = P(
    parser.Type ~ ValueId ~ ":" ~ parser.Type
      ~ parser.OptionalAttributes
  ).map(
    (
        x: Attribute,
        y: String,
        z: Attribute,
        w: DictType[String, Attribute]
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
    override val operands: ListType[Value[Attribute]],
    override val successors: ListType[Block],
    results_types: ListType[Attribute],
    override val regions: ListType[Region],
    override val dictionaryProperties: DictType[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute]
) extends RegisteredOperation(
      name = "subop.set_result",
      operands,
      successors,
      results_types,
      regions,
      dictionaryProperties,
      dictionaryAttributes
    ) {

  override def custom_verify(): Unit = (
    operands.length,
    successors.length,
    results.length,
    regions.length,
    dictionaryProperties.size
  ) match {
    case (1, 0, 0, 0, 0) =>
      dictionaryAttributes.get("result_id") match {
        case Some(x) =>
          x match {
            case _: IntegerAttr =>
            case _ =>
              throw new VerifyException(
                "SetResultOp Operation's 'result_id' must be of type IntegerAttr."
              )
          }
        case _ =>
          throw new VerifyException(
            "SetResultOp Operation's 'result_id' must be of type IntegerAttr."
          )
      }
  }

}

val SubOperatorOps: Dialect =
  new Dialect(
    operations = Seq(SetResultOp),
    attributes = Seq(ResultTable)
  )

// !subop.result_table<[avg_disc$0 : !db.decimal<31, 21>, count_order$0 : i64]>

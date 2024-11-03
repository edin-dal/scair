package scair.dialects.LingoDB.SubOperatorOps

import fastparse._
import scair.dialects.builtin._
import scala.collection.immutable
import scala.collection.mutable
import scair.{Parser, AttrParser}
import scair.Parser.{whitespace, BareId, ValueId, Type, DictionaryAttribute}
import scair.ir._
import scair.exceptions.VerifyException

////////////////
// ATTRIBUTES //
////////////////

// ==------------== //
//   StateMembers   //
// ==------------== //

object StateMembers extends DialectAttribute {
  override def name: String = "subop.state_members"
  override def parse[$: P] = P(
    "[" ~ (BareId.map(StringData(_)) ~ ":" ~ Type).rep(0, sep = ",") ~ "]"
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

object ResultTable extends DialectAttribute {
  override def name: String = "subop.result_table"
  override def parse[$: P] = P(
    "<" ~ StateMembers.parse ~ ">"
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

object SetResultOp extends DialectOperation {
  override def name: String = "subop.set_result"
  override def factory = SetResultOp.apply

  // ==--- Custom Parsing ---== //
  override def parse[$: P](
      resNames: Seq[String],
      parser: Parser
  ): P[Operation] = P(
    Type ~ ValueId ~ ":" ~ Type
      ~ DictionaryAttribute.?.map(Parser.optionlessSeq)
  ).map(
    (
        x: Attribute,
        y: String,
        z: Attribute,
        w: Seq[(String, Attribute)]
    ) =>
      parser.verifyCustomOp(
        opGen = factory,
        opName = name,
        operandNames = Seq(y),
        operandTypes = Seq(z),
        dictAttrs = w :+ ("result_id", x)
      )
  )
  // ==----------------------== //
}

case class SetResultOp(
    override val operands: ListType[Value[Attribute]],
    override val successors: ListType[Block],
    override val results: ListType[Value[Attribute]],
    override val regions: ListType[Region],
    override val dictionaryProperties: DictType[String, Attribute],
    override val dictionaryAttributes: DictType[String, Attribute]
) extends RegisteredOperation(name = "subop.set_result") {

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

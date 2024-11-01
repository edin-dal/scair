package scair.clair

import fastparse._
import fastparse.internal.Util

import scala.collection.mutable
import scala.annotation.tailrec
import scala.annotation.switch
import scala.util.{Try, Success, Failure}
import java.lang.Integer.parseInt

// ░█████╗░ ██╗░░░░░ ░█████╗░ ██╗ ██████╗░
// ██╔══██╗ ██║░░░░░ ██╔══██╗ ██║ ██╔══██╗
// ██║░░╚═╝ ██║░░░░░ ███████║ ██║ ██████╔╝
// ██║░░██╗ ██║░░░░░ ██╔══██║ ██║ ██╔══██╗
// ╚█████╔╝ ███████╗ ██║░░██║ ██║ ██║░░██║
// ░╚════╝░ ╚══════╝ ╚═╝░░╚═╝ ╚═╝ ╚═╝░░╚═╝

// ██████╗░ ░█████╗░ ██████╗░ ░██████╗ ███████╗ ██████╗░
// ██╔══██╗ ██╔══██╗ ██╔══██╗ ██╔════╝ ██╔════╝ ██╔══██╗
// ██████╔╝ ███████║ ██████╔╝ ╚█████╗░ █████╗░░ ██████╔╝
// ██╔═══╝░ ██╔══██║ ██╔══██╗ ░╚═══██╗ ██╔══╝░░ ██╔══██╗
// ██║░░░░░ ██║░░██║ ██║░░██║ ██████╔╝ ███████╗ ██║░░██║
// ╚═╝░░░░░ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ╚═════╝░ ╚══════╝ ╚═╝░░╚═╝

/*≡≡=---=≡≡≡=---=≡≡*\
||     CONTEXT     ||
\*≡==---==≡==---==≡*/

class ParseCTX() {
  val typeCTX: DictType[String, RegularType] = DictType()
  val dialectCTX: DictType[String, DialectDef] = DictType()

  def typeInCTX(name: String): RegularType = {
    if (!typeCTX.contains(name)) {
      throw new Exception(s"Type ${name} used but not defined.")
    }
    typeCTX(name)
  }

  def addCTXtype(name: String, typ: RegularType): Unit = {
    if (typeCTX.contains(name)) {
      throw new Exception(s"Type ${name} already defined.")
    } else {
      typeCTX(name) = typ
    }
  }

  def dialectInCTX(name: String): Boolean = {
    dialectCTX.contains(name)
  }

  def addToCTXdialect(
      name: String,
      ops: ListType[OperationDef],
      attrs: ListType[AttributeDef]
  ): Unit = {
    if (dialectCTX.contains(name)) {
      dialectCTX(name).operations ++= ops
      dialectCTX(name).attributes ++= attrs
    } else {
      dialectCTX(name) = DialectDef(
        name = name,
        attributes = attrs,
        operations = ops
      )
    }
  }

}

class ClairParser() {

  implicit val ctx: ParseCTX = new ParseCTX

  def parseThis[A](
      text: String,
      pattern: P[_] => P[A] = { (x: P[_]) =>
        ClairParser.EntryPoint(x, ctx)
      }
  ): Parsed[A] = {
    return parse(text, pattern)
  }
}

object ClairParser {

  implicit val whitespace: Whitespace = { implicit ctx: P[_] =>
    val input = ctx.input
    val startIndex = ctx.index
    @tailrec def rec(current: Int, state: Int): ParsingRun[Unit] = {
      if (!input.isReachable(current)) {
        if (state == 0 || state == 1) ctx.freshSuccessUnit(current)
        else ctx.freshSuccessUnit(current - 1)
      } else {
        val currentChar = input(current)
        (state: @switch) match {
          case 0 =>
            (currentChar: @switch) match {
              case ' ' | '\t' | '\n' | '\r' => rec(current + 1, state)
              case '/'                      => rec(current + 1, state = 2)
              case _                        => ctx.freshSuccessUnit(current)
            }
          case 1 =>
            rec(current + 1, state = if (currentChar == '\n') 0 else state)
          case 2 =>
            (currentChar: @switch) match {
              case '/' => rec(current + 1, state = 1)
              case _   => ctx.freshSuccessUnit(current - 1)
            }
        }
      }
    }
    rec(current = ctx.index, state = 0)
  }

  def E[$: P](action: => Unit) = {
    action
    Pass(())
  }

  /*≡≡=---=≡≡≡=---=≡≡*\
  ||   ENTRY POINT   ||
  \*≡==----=≡=----==≡*/

  def EntryPoint[$: P](implicit
      ctx: ParseCTX
  ): P[DictType[String, DialectDef]] =
    P(Start ~ TypeDef.? ~ AttributeParser.? ~ OperationParser.? ~ End).map(_ =>
      ctx.dialectCTX
    )

  /*≡≡=---=≡≡≡=---=≡≡*\
  ||      TYPES      ||
  \*≡==----=≡=----==≡*/

  def TypeDef[$: P](implicit ctx: ParseCTX): P[Unit] =
    P((RegularTypeP).rep(0))

  def RegularTypeP[$: P](implicit ctx: ParseCTX): P[Unit] =
    P("{" ~ "type" ~/ BareId ~/ "=" ~/ DialectRefName ~/ "." ~/ BareId ~ "}")
      .map((x, y, z) => ctx.addCTXtype(x, RegularType(y, z)))

  /*≡≡=---=≡≡≡=---=≡≡*\
  ||      BASIC      ||
  \*≡==----=≡=----==≡*/
  // integer ::= [0..9]*
  // ValueDef ::= identifier ":" Type
  // DictDef ::= identifier "=" Type
  // Type ::= identifier
  //

  def Digit[$: P] = P(CharIn("0-9").!)

  def Letter[$: P] = P(CharIn("a-zA-Z").!)

  def BareId[$: P] = P(
    (Letter | "_") ~~ (Letter | Digit | CharIn("_$")).rep
  ).!

  def DialectRefName[$: P] = P(
    (CharIn("a-zA-Z") ~~ CharsWhileIn("a-zA-Z0-9_")).!
  )

  def DecimalLiteral[$: P] =
    P(Digit.rep(1).!).map((literal: String) => parseInt(literal))

  def ConstraintP[$: P](
      sign: String
  )(implicit ctx: ParseCTX): P[(String, ConstraintDef)] =
    P(
      (BareId ~ ":" ~ TypeParser ~ ("|" ~ TypeParser).rep(0)).map((x, y, z) =>
        (x, Any(z :+ y))
      ) |
        (BareId ~ sign ~ TypeParser).map((x, y) => (x, Base(y))) |
        (BareId ~ "==" ~ TypeParser).map((x, y) => (x, Equal(y)))
    )

  def ValueDef[$: P](implicit ctx: ParseCTX): P[(String, ConstraintDef)] =
    P(ConstraintP(":"))

  def DictDef[$: P](implicit ctx: ParseCTX): P[(String, ConstraintDef)] =
    P(ConstraintP("="))

  def TypeParser[$: P](implicit ctx: ParseCTX): P[Type] =
    P(BareId).map((x) => ctx.typeInCTX(x))

  /*≡≡=---=≡≡≡=---=≡≡*\
  ||    OPERATION    ||
  \*≡==----=≡=----==≡*/

  // Operation ::=
  //   OperationInput* "-"* identifier "->" identifier "." identifier

  def OperationParser[$: P](implicit ctx: ParseCTX): P[OperationDef] =
    P(
      OperationInputParser ~
        "-".rep(1) ~ BareId ~
        "->" ~ BareId ~~ "." ~~ DialectRefName
    ).map((x) =>
      val op =
        OperationDef(
          s"${x._8}.${x._9}",
          x._7,
          x._1,
          x._2,
          x._3,
          x._4,
          x._5,
          x._6
        )
      ctx.addToCTXdialect(x._8, ListType(op), ListType())
      op
    )

  /*≡≡=---=≡≡≡=---=≡≡*\
  ||    ATTRIBUTE    ||
  \*≡==----=≡=----==≡*/

  // Attribute ::=
  //   AttributeInput* "-"* identifier "->" identifier "." identifier

  // OperationInput ::=
  //     OperandInput     |
  //     TypeInput        |
  //     DataInput

  // OperandInput    ::= "->" "operands" "[" ValueDef* "]"
  // TypeInput       ::= "type"
  // DataInput       ::= "data"

  def AttributeParser[$: P](implicit ctx: ParseCTX): P[AttributeDef] =
    P(
      AttributeInputParser ~
        "-".rep(1) ~ BareId ~
        "->" ~ BareId ~~ "." ~~ DialectRefName
    ).map((x) =>
      val attr = AttributeDef(s"${x._5}.${x._6}", x._4, x._3, x._1 + x._2)
      ctx.addToCTXdialect(x._5, ListType(), ListType(attr))
      attr
    )

  def AttributeInputParser[$: P](implicit
      ctx: ParseCTX
  ): P[(Int, Int, Seq[OperandDef])] =
    P(
      TypeInput.?.map(_.getOrElse(0)) ~
        DataInput.?.map(_.getOrElse(0)) ~
        OperandsInput.rep(min = 0, max = 1).map(_.flatten)
    )

  def TypeInput[$: P]: P[Int] =
    P("->" ~ "type").map(_ => 1)

  def DataInput[$: P]: P[Int] =
    P("->" ~ "data").map(_ => 2)

  /*≡=---=≡≡≡=---=≡*\
  ||     INPUT     ||
  \*≡==---=≡=---==≡*/

  // OperandInput    ::= "->" "operands" "[" ValueDef* "]"
  // ResultsInput    ::= "->" "results" "[" ValueDef* "]"
  // RegionsInput    ::= "->" "regions" "[" integer "]"
  // SuccessorsInput ::= "->" "successors" "[" integer "]"
  // PropertiesInput ::= "->" "properties" "[" DictDef* "]"
  // AttributesInput ::= "->" "attributes" "[" DictDef* "]"

  def OperationInputParser[$: P](implicit ctx: ParseCTX): P[
    (
        Seq[OperandDef],
        Seq[ResultDef],
        RegionDef,
        SuccessorDef,
        Seq[OpPropertyDef],
        Seq[OpAttributeDef]
    )
  ] =
    P(
      OperandsInput.rep(min = 0, max = 1).map(_.flatten) ~
        ResultsInput.rep(min = 0, max = 1).map(_.flatten) ~
        RegionsInput.?.map(_.getOrElse(RegionDef(0))) ~
        SuccessorsInput.?.map(_.getOrElse(SuccessorDef(0))) ~
        OpPropertiesInput.rep(min = 0, max = 1).map(_.flatten) ~
        OpAttributesInput.rep(min = 0, max = 1).map(_.flatten)
    )

  def OperandsInput[$: P](implicit ctx: ParseCTX): P[Seq[OperandDef]] =
    P(
      "->" ~ "operands" ~ "[" ~ ValueDef
        .map(OperandDef(_, _))
        .rep(1, sep = ",") ~ "]"
    )

  def ResultsInput[$: P](implicit ctx: ParseCTX): P[Seq[ResultDef]] =
    P(
      "->" ~ "results" ~ "[" ~ ValueDef
        .map(ResultDef(_, _))
        .rep(1, sep = ",") ~ "]"
    )

  def RegionsInput[$: P]: P[RegionDef] =
    P("->" ~ "regions" ~ "[" ~ DecimalLiteral.map(RegionDef(_)) ~ "]")

  def SuccessorsInput[$: P]: P[SuccessorDef] =
    P("->" ~ "successors" ~ "[" ~ DecimalLiteral.map(SuccessorDef(_)) ~ "]")

  def OpPropertiesInput[$: P](implicit ctx: ParseCTX): P[Seq[OpPropertyDef]] =
    P(
      "->" ~ "properties" ~ "[" ~ DictDef
        .map(OpPropertyDef(_, _))
        .rep(1, sep = ",") ~ "]"
    )

  def OpAttributesInput[$: P](implicit ctx: ParseCTX): P[Seq[OpAttributeDef]] =
    P(
      "->" ~ "attributes" ~ "[" ~ DictDef
        .map(OpAttributeDef(_, _))
        .rep(1, sep = ",") ~ "]"
    )
}

object Main123 {
  def main(args: Array[String]): Unit = {
    val iinput =
      "-> operands   [name:type]\n" +
        "-> results    [name:type]\n" +
        "-> regions    [0]\n" +
        "-> successors [0]\n" +
        "-> properties [name=type]\n" +
        "-> attributes [name=type]\n" +
        "----------------- NameOp\n" +
        "-> dialect.name"

    val iinput2 =
      "-> type\n" +
        "-> data\n" +
        "-> operands   [name:type]\n" +
        "----------------- NameOp\n" +
        "-> dialect.name"

    val input3 =
      "{ type tt = dialect.name1 }\n" +
        "{ type gg = dialect.name2 }\n"

    val input4 =
      "-> type\n" +
        "----------------- NameAttr\n" +
        "-> dialect.name1"

    val input5 =
      "-> operands   [map:gg, map2==gg, map3 : tt | gg]\n" +
        "----------------- NameOp\n" +
        "-> dialect.name2"

    val input6 =
      "{ type tt = dialect.name1 }\n" +
        "{ type gg = dialect.name2 }\n" +
        "-> type\n" +
        "----------------- NameAttr\n" +
        "-> dialect.name1\n" +
        "-> operands   [map:gg, map2==gg, map3 : tt | gg]\n" +
        "----------------- NameOp\n" +
        "-> dialect.name2"

    val parser = new ClairParser

    // println(parse(iinput, OperationParser(_)))
    // println(parse(iinput2, AttributeParser(_)))
    // println("----Types----")
    // println(parse(input3, TypeDef(_)))
    // println("----Attributes----")
    // println(parse(input4, AttributeParser(_)))
    // println("----Operations----")
    // println(parse(input5, OperationParser(_)))
    println("----WholeThing----")
    val parsed = parser.parseThis(input6)
    println(parsed)
  }
}

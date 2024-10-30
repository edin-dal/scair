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

  // TODO: type parsing and type context as well
  // TODO: operation & attribute name context (can just be one)
  // TODO:

  /*≡≡=---=≡≡=---=≡≡*\
  ||    CONTEXTS    ||
  \*≡==---=≡≡=---==≡*/

  val typeCTX: DictType[String, RegularType] = DictType()
  // val anonCTX: DictType[String, AnonType] = DictType()
  val dialectCTX: DictType[String, Dialect] = DictType()

  def typeInCTX(name: String): RegularType = {
    if (!typeCTX.contains(name)) {
      throw new Exception(s"Type ${name} used but not defined.")
    }
    typeCTX(name)
  }

  // def anonInCTX(name: String): AnonType = {
  //   if (!anonCTX.contains(name)) {
  //     throw new Exception(s"Anonymous type ${name} used but not defined.")
  //   }
  //   anonCTX(name)
  // }

  def addCTXtype(name: String, typ: RegularType): Unit = {
    if (typeCTX.contains(name)) {
      throw new Exception(s"Type ${name} already defined.")
    } else {
      typeCTX(name) = typ
    }
  }

  // def addCTXanon(name: String, typ: AnonType): Unit = {
  //   if (anonCTX.contains(name)) {
  //     throw new Exception(s"Anonymous type ${name} already defined.")
  //   } else {
  //     anonCTX(name) = typ
  //   }
  // }

  /*≡≡=---=≡≡≡=---=≡≡*\
  ||   ENTRY POINT   ||
  \*≡==----=≡=----==≡*/

  def EntryPoint[$: P]: P[DictType[String, Dialect]] =
    P(
      TypeDef ~ E(println("1")) ~ AttributeParser ~ E(
        println("2")
      ) ~ OperationParser ~ E(println("3")) ~ End
    ).map(_ => dialectCTX)

  /*≡≡=---=≡≡≡=---=≡≡*\
  ||      TYPES      ||
  \*≡==----=≡=----==≡*/

  def TypeDef[$: P]: P[Unit] =
    P((RegularTypeP).rep(0))

  def RegularTypeP[$: P]: P[Unit] =
    P("{" ~ "type" ~/ BareId ~/ "=" ~/ DialectRefName ~/ "." ~/ BareId ~ "}")
      .map((x, y, z) => addCTXtype(x, RegularType(y, z)))

  // def AnonTypeP[$: P]: P[Unit] =
  //   P("anon " ~ BareId ~ ":" ~ DialectRefName ~ "." ~ BareId).map((x, y, z) =>
  //     addCTXanon(x, AnonType(x, RegularType(y, z)))
  //   )

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

  def ConstraintP[$: P](sign: String): P[(String, Constraint)] =
    P(
      (BareId ~ ":" ~ TypeParser ~ ("|" ~ TypeParser).rep(0)).map((x, y, z) =>
        (x, Any(z :+ y))
      ) |
        (BareId ~ sign ~ TypeParser).map((x, y) => (x, Base(y))) |
        (BareId ~ "==" ~ TypeParser).map((x, y) => (x, Equal(y)))
        // (BareId ~ "::" ~ BareId ~ ":" ~ "[" ~ ConstraintP ~ "]").map(
        //   (x, y, z) => (x, Anon(y))
        // )
    )

  def ValueDef[$: P]: P[(String, Constraint)] =
    P(ConstraintP(":"))

  def DictDef[$: P]: P[(String, Constraint)] =
    P(ConstraintP("="))

  def TypeParser[$: P]: P[Type] =
    P(BareId).map((x) => typeInCTX(x)
    // Try(typeInCTX(x)) match {
    //   case Success(a) => a
    //   case Failure(_) => anonInCTX(x)
    // }
    )

  /*≡≡=---=≡≡≡=---=≡≡*\
  ||    OPERATION    ||
  \*≡==----=≡=----==≡*/

  // Operation ::=
  //   OperationInput* "-"* identifier "->" identifier "." identifier

  def OperationParser[$: P]: P[Operation] =
    P(
      OperationInputParser ~
        "-".rep(1) ~ BareId ~
        "->" ~ BareId ~~ "." ~~ DialectRefName
    ).map((x) =>
      val op =
        Operation(s"${x._8}.${x._9}", x._7, x._1, x._2, x._3, x._4, x._5, x._6)

      if (dialectCTX.contains(x._8)) {
        dialectCTX(x._8).operations += op
      } else {
        dialectCTX(x._8) = Dialect(
          name = x._8,
          operations = ListType(op)
        )
      }

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

  def AttributeParser[$: P]: P[Attribute] =
    P(
      AttributeInputParser ~
        "-".rep(1) ~ BareId ~
        "->" ~ BareId ~~ "." ~~ DialectRefName
    ).map((x) =>
      val attr = Attribute(s"${x._5}.${x._6}", x._4, x._3, x._1 + x._2)

      if (dialectCTX.contains(x._5)) {
        dialectCTX(x._5).attributes += attr
      } else {
        dialectCTX(x._5) = Dialect(
          name = x._5,
          attributes = ListType(attr)
        )
      }

      attr
    )

  def AttributeInputParser[$: P]: P[(Int, Int, Seq[Operand])] =
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

  def OperationInputParser[$: P]: P[
    (
        Seq[Operand],
        Seq[Result],
        Region,
        Successor,
        Seq[OpProperty],
        Seq[OpAttribute]
    )
  ] =
    P(
      OperandsInput.rep(min = 0, max = 1).map(_.flatten) ~
        ResultsInput.rep(min = 0, max = 1).map(_.flatten) ~
        RegionsInput.?.map(_.getOrElse(Region(0))) ~
        SuccessorsInput.?.map(_.getOrElse(Successor(0))) ~
        OpPropertiesInput.rep(min = 0, max = 1).map(_.flatten) ~
        OpAttributesInput.rep(min = 0, max = 1).map(_.flatten)
    )

  def OperandsInput[$: P]: P[Seq[Operand]] =
    P(
      "->" ~ "operands" ~ "[" ~ ValueDef
        .map(Operand(_, _))
        .rep(1, sep = ",") ~ "]"
    )

  def ResultsInput[$: P]: P[Seq[Result]] =
    P(
      "->" ~ "results" ~ "[" ~ ValueDef
        .map(Result(_, _))
        .rep(1, sep = ",") ~ "]"
    )

  def RegionsInput[$: P]: P[Region] =
    P("->" ~ "regions" ~ "[" ~ DecimalLiteral.map(Region(_)) ~ "]")

  def SuccessorsInput[$: P]: P[Successor] =
    P("->" ~ "successors" ~ "[" ~ DecimalLiteral.map(Successor(_)) ~ "]")

  def OpPropertiesInput[$: P]: P[Seq[OpProperty]] =
    P(
      "->" ~ "properties" ~ "[" ~ DictDef
        .map(OpProperty(_, _))
        .rep(1, sep = ",") ~ "]"
    )

  def OpAttributesInput[$: P]: P[Seq[OpAttribute]] =
    P(
      "->" ~ "attributes" ~ "[" ~ DictDef
        .map(OpAttribute(_, _))
        .rep(1, sep = ",") ~ "]"
    )

  def parseThis[A](
      text: String,
      pattern: P[_] => P[A] = { (x: P[_]) =>
        EntryPoint(x)
      }
  ): Parsed[A] = {
    return parse(text, pattern)
  }

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

    // println(parse(iinput, OperationParser(_)))
    // println(parse(iinput2, AttributeParser(_)))
    // println("----Types----")
    // println(parse(input3, TypeDef(_)))
    // println("----Attributes----")
    // println(parse(input4, AttributeParser(_)))
    // println("----Operations----")
    // println(parse(input5, OperationParser(_)))
    println("----WholeThing----")
    println(parse(input6, EntryPoint(_)))
    println(typeCTX)
  }

}

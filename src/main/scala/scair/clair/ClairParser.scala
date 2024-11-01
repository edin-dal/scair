// package scair.clair

// import fastparse._
// import fastparse.internal.Util
// import scala.util.control.NoStackTrace

// import scala.collection.mutable
// import scala.annotation.tailrec
// import scala.annotation.switch
// import scala.util.{Try, Success, Failure}
// import java.lang.Integer.parseInt
// import scair.clair.ClairCustomException

// // ░█████╗░ ██╗░░░░░ ░█████╗░ ██╗ ██████╗░
// // ██╔══██╗ ██║░░░░░ ██╔══██╗ ██║ ██╔══██╗
// // ██║░░╚═╝ ██║░░░░░ ███████║ ██║ ██████╔╝
// // ██║░░██╗ ██║░░░░░ ██╔══██║ ██║ ██╔══██╗
// // ╚█████╔╝ ███████╗ ██║░░██║ ██║ ██║░░██║
// // ░╚════╝░ ╚══════╝ ╚═╝░░╚═╝ ╚═╝ ╚═╝░░╚═╝

// // ██████╗░ ░█████╗░ ██████╗░ ░██████╗ ███████╗ ██████╗░
// // ██╔══██╗ ██╔══██╗ ██╔══██╗ ██╔════╝ ██╔════╝ ██╔══██╗
// // ██████╔╝ ███████║ ██████╔╝ ╚█████╗░ █████╗░░ ██████╔╝
// // ██╔═══╝░ ██╔══██║ ██╔══██╗ ░╚═══██╗ ██╔══╝░░ ██╔══██╗
// // ██║░░░░░ ██║░░██║ ██║░░██║ ██████╔╝ ███████╗ ██║░░██║
// // ╚═╝░░░░░ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ╚═════╝░ ╚══════╝ ╚═╝░░╚═╝

// /*≡≡=---=≡≡≡=---=≡≡*\
// ||     CONTEXT     ||
// \*≡==---==≡==---==≡*/

// class ParseCTX() {
//   val typeCTX: DictType[String, RegularType] = DictType()
//   val dialectCTX: DictType[String, DialectDef] = DictType()
//   val namingCTX: ListType[String] = ListType()

//   def typeInCTX(name: String): RegularType = {
//     if (!typeCTX.contains(name)) {
//       throw new ClairCustomException(
//         "ParseCTXException",
//         s"Type ${name} used but not defined."
//       )
//     }
//     typeCTX(name)
//   }

//   def addCTXtype(name: String, typ: RegularType): Unit = {
//     if (typeCTX.contains(name)) {
//       throw new ClairCustomException(
//         "ParseCTXException",
//         s"Type ${name} already defined."
//       )
//     } else {
//       typeCTX(name) = typ
//     }
//   }

//   def addToCTXdialect(
//       name: String,
//       ops: ListType[OperationDef],
//       attrs: ListType[AttributeDef]
//   ): Unit = {
//     if (dialectCTX.contains(name)) {
//       dialectCTX(name).operations ++= ops
//       dialectCTX(name).attributes ++= attrs
//     } else {
//       dialectCTX(name) = DialectDef(
//         name = name,
//         attributes = attrs,
//         operations = ops
//       )
//     }
//   }

//   def checkNameInCTXandAdd(name: String): Unit = {
//     if (namingCTX.contains(name)) {
//       throw new ClairCustomException(
//         "ParseCTXException",
//         s"Member '$name' already defined in the file."
//       )
//     } else {
//       namingCTX += name
//     }
//   }
// }

// class LocalCTX() {
//   val namingCTX: ListType[String] = ListType()

//   def checkNameInCTXandAdd(name: String): Unit = {
//     if (namingCTX.contains(name)) {
//       throw new ClairCustomException(
//         "LocalCTXException",
//         s"Field '$name' alreadyn defined within one member's local context."
//       )
//     } else {
//       namingCTX += name
//     }
//   }
// }

// class ClairParser() {

//   implicit val ctx: ParseCTX = new ParseCTX

//   def parseThis[A](
//       text: String,
//       pattern: P[_] => P[A] = { (x: P[_]) =>
//         ClairParser.EntryPoint(x, ctx)
//       }
//   ): Parsed[A] = {
//     return parse(text, pattern)
//   }
// }

// object ClairParser {

//   implicit val whitespace: Whitespace = { implicit ctx: P[_] =>
//     val input = ctx.input
//     val startIndex = ctx.index
//     @tailrec def rec(current: Int, state: Int): ParsingRun[Unit] = {
//       if (!input.isReachable(current)) {
//         if (state == 0 || state == 1) ctx.freshSuccessUnit(current)
//         else ctx.freshSuccessUnit(current - 1)
//       } else {
//         val currentChar = input(current)
//         (state: @switch) match {
//           case 0 =>
//             (currentChar: @switch) match {
//               case ' ' | '\t' | '\n' | '\r' => rec(current + 1, state)
//               case '/'                      => rec(current + 1, state = 2)
//               case _                        => ctx.freshSuccessUnit(current)
//             }
//           case 1 =>
//             rec(current + 1, state = if (currentChar == '\n') 0 else state)
//           case 2 =>
//             (currentChar: @switch) match {
//               case '/' => rec(current + 1, state = 1)
//               case _   => ctx.freshSuccessUnit(current - 1)
//             }
//         }
//       }
//     }
//     rec(current = ctx.index, state = 0)
//   }

//   def E[$: P](action: => Unit) = {
//     action
//     Pass(())
//   }

//   /*≡≡=---=≡≡≡=---=≡≡*\
//   ||   ENTRY POINT   ||
//   \*≡==----=≡=----==≡*/

//   def EntryPoint[$: P](implicit
//       ctx: ParseCTX
//   ): P[DictType[String, DialectDef]] =
//     P(Start ~ TypeDef.? ~ AttributeParser.? ~ OperationParser.? ~ End).map(_ =>
//       ctx.dialectCTX
//     )

//   /*≡≡=---=≡≡≡=---=≡≡*\
//   ||      TYPES      ||
//   \*≡==----=≡=----==≡*/

//   def TypeDef[$: P](implicit ctx: ParseCTX): P[Unit] =
//     P((RegularTypeP).rep(0))

//   def RegularTypeP[$: P](implicit ctx: ParseCTX): P[Unit] =
//     P("{" ~ "type" ~/ BareId ~/ "=" ~/ DialectRefName ~/ "." ~/ BareId ~ "}")
//       .map((x, y, z) => ctx.addCTXtype(x, RegularType(y, z)))

//   /*≡≡=---=≡≡≡=---=≡≡*\
//   ||      BASIC      ||
//   \*≡==----=≡=----==≡*/
//   // integer ::= [0..9]*
//   // ValueDef ::= identifier ":" Type
//   // DictDef ::= identifier "=" Type
//   // Type ::= identifier
//   //

//   def Digit[$: P] = P(CharIn("0-9").!)

//   def Letter[$: P] = P(CharIn("a-zA-Z").!)

//   def BareId[$: P] = P(
//     (Letter | "_") ~~ (Letter | Digit | CharIn("_$")).rep
//   ).!

//   def DialectRefName[$: P] = P(
//     (CharIn("a-zA-Z") ~~ CharsWhileIn("a-zA-Z0-9_")).!
//   )

//   def DecimalLiteral[$: P] =
//     P(Digit.rep(1).!).map((literal: String) => parseInt(literal))

//   def ConstraintP[$: P](
//       sign: String
//   )(implicit ctx: ParseCTX): P[(String, ConstraintDef)] =
//     P(
//       (BareId ~ sign ~ TypeParser ~ "|" ~ TypeParser ~
//         ("|" ~ TypeParser).rep(0))
//         .map((x, y1, y2, z) => (x, AnyOf(z :+ y1 :+ y2))) |
//         (BareId ~ sign ~ TypeParser).map((x, y) => (x, Base(y))) |
//         (BareId ~ "==" ~ TypeParser).map((x, y) => (x, Equal(y)))
//     )

//   def ValueDef[$: P](implicit ctx: ParseCTX): P[(String, ConstraintDef)] =
//     P(ConstraintP(":"))

//   def DictDef[$: P](implicit ctx: ParseCTX): P[(String, ConstraintDef)] =
//     P(ConstraintP("="))

//   def TypeParser[$: P](implicit ctx: ParseCTX): P[Type] =
//     P(BareId).map((x) => ctx.typeInCTX(x))

//   /*≡≡=---=≡≡≡=---=≡≡*\
//   ||    OPERATION    ||
//   \*≡==----=≡=----==≡*/

//   // Operation ::=
//   //   OperationInput* "-"* identifier "->" identifier "." identifier

//   def OperationParser[$: P](implicit ctx: ParseCTX): P[OperationDef] =
//     P(
//       OperationInputParser ~
//         "-".rep(1) ~ BareId ~
//         "->" ~ BareId ~~ "." ~~ DialectRefName
//     ).map((x) =>
//       val opName = s"${x._8}.${x._9}"
//       ctx.checkNameInCTXandAdd(opName)
//       val op =
//         OperationDef(
//           opName,
//           x._7,
//           x._1,
//           x._2,
//           x._3,
//           x._4,
//           x._5,
//           x._6
//         )
//       ctx.addToCTXdialect(x._8, ListType(op), ListType())
//       op
//     )

//   /*≡≡=---=≡≡≡=---=≡≡*\
//   ||    ATTRIBUTE    ||
//   \*≡==----=≡=----==≡*/

//   // Attribute ::=
//   //   AttributeInput* "-"* identifier "->" identifier "." identifier

//   // OperationInput ::=
//   //     OperandInput     |
//   //     TypeInput        |
//   //     DataInput

//   // OperandInput    ::= "->" "operands" "[" ValueDef* "]"
//   // TypeInput       ::= "type"
//   // DataInput       ::= "data"

//   def AttributeParser[$: P](implicit ctx: ParseCTX): P[AttributeDef] =
//     P(
//       AttributeInputParser ~
//         "-".rep(1) ~ BareId ~
//         "->" ~ BareId ~~ "." ~~ DialectRefName
//     ).map((x) =>
//       val attrName = s"${x._5}.${x._6}"
//       ctx.checkNameInCTXandAdd(attrName)
//       val attr = AttributeDef(attrName, x._4, x._3, x._1 + x._2)
//       ctx.addToCTXdialect(x._5, ListType(), ListType(attr))
//       attr
//     )

//   def AttributeInputParser[$: P](implicit
//       ctx: ParseCTX
//   ): P[(Int, Int, Seq[OperandDef])] = {
//     val lctx: LocalCTX = new LocalCTX
//     P(
//       TypeInput.?.map(_.getOrElse(0)) ~
//         DataInput.?.map(_.getOrElse(0)) ~
//         OperandsInput(lctx).rep(min = 0, max = 1).map(_.flatten)
//     )
//   }

//   def TypeInput[$: P]: P[Int] =
//     P("->" ~ "type").map(_ => 1)

//   def DataInput[$: P]: P[Int] =
//     P("->" ~ "data").map(_ => 2)

//   /*≡=---=≡≡≡=---=≡*\
//   ||     INPUT     ||
//   \*≡==---=≡=---==≡*/

//   // OperandInput    ::= "->" "operands" "[" ValueDef* "]"
//   // ResultsInput    ::= "->" "results" "[" ValueDef* "]"
//   // RegionsInput    ::= "->" "regions" "[" integer "]"
//   // SuccessorsInput ::= "->" "successors" "[" integer "]"
//   // PropertiesInput ::= "->" "properties" "[" DictDef* "]"
//   // AttributesInput ::= "->" "attributes" "[" DictDef* "]"

//   def OperationInputParser[$: P](implicit ctx: ParseCTX): P[
//     (
//         Seq[OperandDef],
//         Seq[ResultDef],
//         RegionDef,
//         SuccessorDef,
//         Seq[OpPropertyDef],
//         Seq[OpAttributeDef]
//     )
//   ] = {
//     val lctx: LocalCTX = new LocalCTX
//     P(
//       OperandsInput(lctx).rep(min = 0, max = 1).map(_.flatten) ~
//         ResultsInput(lctx).rep(min = 0, max = 1).map(_.flatten) ~
//         RegionsInput.?.map(_.getOrElse(RegionDef(0))) ~
//         SuccessorsInput.?.map(_.getOrElse(SuccessorDef(0))) ~
//         OpPropertiesInput(lctx).rep(min = 0, max = 1).map(_.flatten) ~
//         OpAttributesInput(lctx).rep(min = 0, max = 1).map(_.flatten)
//     )
//   }

//   def OperandsInput[$: P](
//       lctx: LocalCTX
//   )(implicit ctx: ParseCTX): P[Seq[OperandDef]] =
//     P(
//       "->" ~ "operands" ~ "[" ~ ValueDef
//         .map((x, y) =>
//           lctx.checkNameInCTXandAdd(x)
//           OperandDef(x, y)
//         )
//         .rep(1, sep = ",") ~ "]"
//     )

//   def ResultsInput[$: P](
//       lctx: LocalCTX
//   )(implicit ctx: ParseCTX): P[Seq[ResultDef]] =
//     P(
//       "->" ~ "results" ~ "[" ~ ValueDef
//         .map((x, y) =>
//           lctx.checkNameInCTXandAdd(x)
//           ResultDef(x, y)
//         )
//         .rep(1, sep = ",") ~ "]"
//     )

//   def RegionsInput[$: P]: P[RegionDef] =
//     P("->" ~ "regions" ~ "[" ~ DecimalLiteral.map(RegionDef(_)) ~ "]")

//   def SuccessorsInput[$: P]: P[SuccessorDef] =
//     P("->" ~ "successors" ~ "[" ~ DecimalLiteral.map(SuccessorDef(_)) ~ "]")

//   def OpPropertiesInput[$: P](
//       lctx: LocalCTX
//   )(implicit ctx: ParseCTX): P[Seq[OpPropertyDef]] =
//     P(
//       "->" ~ "properties" ~ "[" ~ DictDef
//         .map((x, y) =>
//           lctx.checkNameInCTXandAdd(x)
//           OpPropertyDef(x, y)
//         )
//         .rep(1, sep = ",") ~ "]"
//     )

//   def OpAttributesInput[$: P](
//       lctx: LocalCTX
//   )(implicit ctx: ParseCTX): P[Seq[OpAttributeDef]] =
//     P(
//       "->" ~ "attributes" ~ "[" ~ DictDef
//         .map((x, y) =>
//           lctx.checkNameInCTXandAdd(x)
//           OpAttributeDef(x, y)
//         )
//         .rep(1, sep = ",") ~ "]"
//     )
// }

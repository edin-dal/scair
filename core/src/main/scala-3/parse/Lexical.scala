package scair.parse

import fastparse.*

import java.lang.Double.parseDouble
import java.lang.Math.pow
import scala.annotation.switch
import scala.annotation.tailrec

/** Whitespace syntax that supports // line-comments, *without* /* */ comments,
  * as is the case in the MLIR Language Spec.
  *
  * It's litteraly fastparse's JavaWhitespace with the /* */ states just erased
  * :)
  */
given whitespace: Whitespace = new Whitespace:
  def apply(ctx: P[?]) =
    val input = ctx.input
    val startIndex = ctx.index
    @tailrec def rec(current: Int, state: Int): ParsingRun[Unit] =
      if !input.isReachable(current) then
        if state == 0 || state == 1 then ctx.freshSuccessUnit(current)
        else ctx.freshSuccessUnit(current - 1)
      else
        val currentChar = input(current)
        (state: @switch) match
          case 0 =>
            (currentChar: @switch) match
              case ' ' | '\t' | '\n' | '\r' => rec(current + 1, state)
              case '/'                      => rec(current + 1, state = 2)
              case _                        => ctx.freshSuccessUnit(current)
          case 1 =>
            rec(current + 1, state = if currentChar == '\n' then 0 else state)
          case 2 =>
            (currentChar: @switch) match
              case '/' => rec(current + 1, state = 1)
              case _   => ctx.freshSuccessUnit(current - 1)
    rec(current = ctx.index, state = 0)

/*≡==--==≡≡≡==--=≡≡*\
||  COMMON SYNTAX  ||
\*≡==---==≡==---==≡*/

// [x] digit     ::= [0-9]
// [x] hex_digit ::= [0-9a-fA-F]
// [x] letter    ::= [a-zA-Z]
// [x] id-punct  ::= [$._-]

// [x] integer-literal ::= decimal-literal | hexadecimal-literal
// [x] decimal-literal ::= digit+
// [x] hexadecimal-literal ::= `0x` hex_digit+
// [x] float-literal ::= [-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?
// [x] string-literal  ::= `"` [^"\n\f\v\r]* `"`

inline val DecDigit = "0-9"
inline val HexDigit = "0-9a-fA-F"

inline def decDigitsP[$: P] = CharsWhileIn(DecDigit)

inline def hexDigitsP[$: P] = CharsWhileIn(HexDigit)

inline val Letter = "a-zA-Z"
inline val IdPunct = "$._\\-"

inline def integerLiteralP[$: P] = hexadecimalLiteralP | decimalLiteralP

def decimalLiteralP[$: P] = (("-" | "+").? ~ decDigitsP).!.map(BigInt(_))

inline def hexadecimalLiteralP[$: P] =
  "0x" ~~ hexDigitsP.!.map(BigInt(_, 16))

private inline def parseFloatNum(float: (String, String)): Double =
  val number = parseDouble(float._1)
  val power = parseDouble(float._2)
  number * pow(10, power)

/** Parses a floating-point number from its string representation. NOTE: This is
  * only a float approximation, and in its current form should not be trusted
  * for precision sensitive applications.
  *
  * @return
  *   float: (String, String)
  */
inline def floatLiteralP[$: P] =
  (
    (CharIn("\\-\\+").? ~~ decDigitsP ~~ "." ~~ decDigitsP).! ~~
      (CharIn("eE") ~~ (CharIn("\\-\\+").? ~~ decDigitsP).!).orElse("0")
  ).map(parseFloatNum(_)) // substituted [0-9]* with [0-9]+

@tailrec
def stringLiteralBisRecP(using
    ctx: ParsingRun[?],
    input: ParserInput,
    str: java.lang.StringBuilder,
)(index: Int): ParsingRun[String] =
  if !input.isReachable(index) then ctx.freshFailure()
  else
    val c0 = input(index)
    (c0: @switch) match
      case '"' => ctx.freshSuccess[String](str.toString(), index + 1)
      case '\n' | 11 | '\f' => Fail("expected '\"' in string literal")
      case '\\'             =>
        val c1 = input(index + 1)
        (c1: @switch) match
          case '"' =>
            str.append('"')
            stringLiteralBisRecP(index + 2)
          case '\\' =>
            str.append('\\')
            stringLiteralBisRecP(index + 2)
          case 'n' =>
            str.append('\n')
            stringLiteralBisRecP(index + 2)
          case 't' =>
            str.append('\t')
            stringLiteralBisRecP(index + 2)
          case 'a' | 'b' | 'c' | 'd' | 'e' | 'f' | 'A' | 'B' | 'C' | 'D' | 'E' |
              'F' | '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' =>
            val c2 = input(index + 2)
            (c2: @switch) match
              case 'a' | 'b' | 'c' | 'd' | 'e' | 'f' | 'A' | 'B' | 'C' | 'D' |
                  'E' | 'F' | '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' |
                  '8' | '9' =>
                str
                  .append(
                    Integer.parseInt(String(Array[Char](c1, c2)), 16).toChar
                  )
                stringLiteralBisRecP(index + 3)
              case _ =>
                Fail("unknown escape in string literal")
          case _ =>
            Fail("unknown escape in string literal")
      case _ =>
        str.append(c0)
        stringLiteralBisRecP(index + 1)

def stringLiteralP(using ctx: ParsingRun[?]): ParsingRun[String] =
  val input = ctx.input
  var index = ctx.index
  if !input.isReachable(index) || input(index) != '"' then ctx.freshFailure()
  else
    stringLiteralBisRecP(using ctx, input, java.lang.StringBuilder())(index + 1)

/*≡==--==≡≡≡==--=≡≡*\
||   IDENTIFIERS   ||
\*≡==---==≡==---==≡*/

// [x] bare-id ::= (letter|[_]) (letter|digit|[_$.])*
// [ ] bare-id-list ::= bare-id (`,` bare-id)*
// [x] value-id ::= `%` suffix-id
// [x] alias-name :: = bare-id
// [x] suffix-id ::= (digit+ | ((letter|id-punct) (letter|id-punct|digit)*))

// [ ] symbol-ref-id ::= `@` (suffix-id | string-literal) // - redundant - (`::` symbol-ref-id)?
// [ ] value-id-list ::= value-id (`,` value-id)*

// // Uses of value, e.g. in an operand list to an operation.
// [x] value-use ::= value-id (`#` decimal-literal)?
// [x] value-use-list ::= value-use (`,` value-use)*

inline def bareIdP[$: P] =
  (
    CharIn(Letter + "_") ~~ CharsWhileIn(Letter + DecDigit + "_$.", min = 0)
  ).!

inline def valueIdP[$: P] = P("%" ~~ suffixIdP.!)

// Alias can't have dots in their names for ambiguity with dialect names.
inline def aliasNameP[$: P] =
  (
    CharIn(Letter + "_") ~~
      (CharsWhileIn(
        Letter + DecDigit + "_$",
        min = 0,
      )) ~~ !"."
  ).!

def suffixIdP[$: P] =
  (
    decDigitsP | CharIn(Letter + IdPunct) ~~ CharsWhileIn(
      Letter + IdPunct + DecDigit,
      min = 0,
    )
  )

inline def symbolRefIdP[$: P] = P("@" ~~ (suffixIdP.! | stringLiteralP))

def operandNameP[$: P] =
  "%" ~~ (suffixIdP ~ ("#" ~~ decDigitsP).?).!

def operandNamesP[$: P] =
  P(operandNameP.rep(sep = ","))

/*≡==--==≡≡≡==--=≡≡*\
||  DIALECT TYPES  ||
\*≡==---==≡==---==≡*/

// [x] - dialect-namespace      ::= bare-id

// [x] - dialect-type           ::= `!` (opaque-dialect-type | pretty-dialect-type)
// [x] - opaque-dialect-type    ::= dialect-namespace dialect-type-body
// [x] - pretty-dialect-type    ::= dialect-namespace `.` pretty-dialect-type-lead-ident dialect-type-body?
// [x] - pretty-dialect-type-lead-ident ::= `[A-Za-z][A-Za-z0-9._]*`

// [x] - dialect-type-body      ::= `<` dialect-type-contents+ `>`
// [x] - dialect-type-contents  ::= dialect-type-body
//                             | `(` dialect-type-contents+ `)`
//                             | `[` dialect-type-contents+ `]`
//                             | `{` dialect-type-contents+ `}`
//                             | [^\[<({\]>)}\0]+

val excludedCharactersDTC: Set[Char] =
  Set('\\', '[', '<', '(', '{', '}', ')', '>', ']', '\u0000')

def notExcludedDTCP[$: P] = P(
  CharPred(char => !excludedCharactersDTC.contains(char))
)

def dialectBareIdP[$: P] = P(
  CharIn(Letter + "_") ~~ CharsWhileIn(Letter + DecDigit + "_$", min = 0)
).!

def dialectNamespaceP[$: P] = P(dialectBareIdP)

def prettyDialectReferenceNameP[$: P] = P(
  (dialectNamespaceP ~ "." ~ prettyDialectTypeOrAttReferenceNameP)
)

def opaqueDialectReferenceNameP[$: P] = P(
  (dialectNamespaceP ~ "<" ~ prettyDialectTypeOrAttReferenceNameP)
)

def dialectReferenceNameP[$: P] = P(
  prettyDialectReferenceNameP | opaqueDialectReferenceNameP
)

def prettyDialectTypeOrAttReferenceNameP[$: P] = P(
  (CharIn("a-zA-Z") ~~ CharsWhileIn("a-zA-Z0-9_")).!
)

/*≡==--==≡≡≡≡==--=≡≡*\
||      BLOCKS      ||
\*≡==---==≡≡==---==≡*/

// [x] - block-id        ::= caret-id
// [x] - caret-id        ::= `^` suffix-id

def blockIdP[$: P] = P(caretIdP)

def caretIdP[$: P] = P("^" ~~/ suffixIdP.!)

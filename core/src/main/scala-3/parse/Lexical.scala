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

inline def DecDigits[$: P] = CharsWhileIn(DecDigit)

inline def HexDigits[$: P] = CharsWhileIn(HexDigit)

inline val Letter = "a-zA-Z"
inline val IdPunct = "$._\\-"

def IntegerLiteral[$: P] = P(HexadecimalLiteral | DecimalLiteral)

def DecimalLiteral[$: P] =
  P(("-" | "+").?.! ~ DecDigits.!).map((sign: String, literal: String) =>
    BigInt(sign + literal)
  )

def HexadecimalLiteral[$: P] =
  P("0x" ~~ HexDigits.!).map((hex: String) => BigInt(hex, 16))

private def parseFloatNum(float: (String, String)): Double =
  val number = parseDouble(float._1)
  val power = parseDouble(float._2)
  return number * pow(10, power)

/** Parses a floating-point number from its string representation. NOTE: This is
  * only a float approximation, and in its current form should not be trusted
  * for precision sensitive applications.
  *
  * @return
  *   float: (String, String)
  */
def FloatLiteral[$: P] = P(
  (CharIn("\\-\\+").? ~~ DecDigits ~~ "." ~~ DecDigits).!
    ~~ (CharIn("eE")
      ~~ (CharIn("\\-\\+").? ~~ DecDigits).!).orElse("0")
).map(parseFloatNum(_)) // substituted [0-9]* with [0-9]+

inline def nonExcludedCharacter(c: Char): Boolean =
  c: @switch match
    case '"' | '\\' => false
    case _          => true

inline def EscapedP[$: P] = P(
  ("\\" ~~ (
    "n" ~~ Pass('\n')
      | "t" ~~ Pass('\t')
      | "\\" ~~ Pass('\\')
      | "\"" ~~ Pass('\"')
      | CharIn("a-fA-F0-9")
        .repX(exactly = 2)
        .!
        .map(Integer.parseInt(_, 16).toChar)
  )).repX.map(chars => String(chars.toArray))
)

def StringLiteral[$: P] = P(
  "\"" ~~/ (CharsWhile(nonExcludedCharacter).! ~~ EscapedP)
    .map(_ + _)
    .repX
    .map(_.mkString) ~~ "\""
)

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

def BareId[$: P] = P(
  CharIn(Letter + "_") ~~ CharsWhileIn(Letter + DecDigit + "_$.", min = 0)
).!

def ValueId[$: P] = P("%" ~~ SuffixId)

// Alias can't have dots in their names for ambiguity with dialect names.
def AliasName[$: P] = P(
  CharIn(Letter + "_") ~~ (CharsWhileIn(
    Letter + DecDigit + "_$",
    min = 0
  )) ~~ !"."
).!

def SuffixId[$: P] = P(
  DecimalLiteral | CharIn(Letter + IdPunct) ~~ CharsWhileIn(
    Letter + IdPunct + DecDigit,
    min = 0
  )
).!

def SymbolRefId[$: P] = P("@" ~~ (SuffixId | StringLiteral))

def ValueUse[$: P] =
  P(ValueId ~ ("#" ~~ DecimalLiteral).?).!.map(_.tail)

def ValueUseList[$: P] =
  P(ValueUse.rep(sep = ","))

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

def notExcludedDTC[$: P] = P(
  CharPred(char => !excludedCharactersDTC.contains(char))
)

def DialectBareId[$: P] = P(
  CharIn(Letter + "_") ~~ CharsWhileIn(Letter + DecDigit + "_$", min = 0)
).!

def DialectNamespace[$: P] = P(DialectBareId)

def PrettyDialectReferenceName[$: P] = P(
  (DialectNamespace ~ "." ~ PrettyDialectTypeOrAttReferenceName)
)

def OpaqueDialectReferenceName[$: P] = P(
  (DialectNamespace ~ "<" ~ PrettyDialectTypeOrAttReferenceName)
)

def DialectReferenceName[$: P] = P(
  PrettyDialectReferenceName | OpaqueDialectReferenceName
)

def PrettyDialectTypeOrAttReferenceName[$: P] = P(
  (CharIn("a-zA-Z") ~~ CharsWhileIn("a-zA-Z0-9_")).!
)

/*≡==--==≡≡≡≡==--=≡≡*\
||      BLOCKS      ||
\*≡==---==≡≡==---==≡*/

// [x] - block-id        ::= caret-id
// [x] - caret-id        ::= `^` suffix-id

def BlockId[$: P] = P(CaretId)

def CaretId[$: P] = P("^" ~~/ SuffixId)

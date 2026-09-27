package scair.parse

import fastparse.*
import fastparse.Parsed.Failure
import scair.ir.*

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||     PARSER INTERFACE     ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

/** The parser, as seen by dialects and derived definitions.
  *
  * Only the IR-aware entry points are abstract; lexical rules and combinators
  * are concrete and dependency-free. The default implementation is
  * [[MLIRParser]].
  */
abstract class Parser:

  def attributeP[$: P]: P[Attribute]
  def typeP[$: P]: P[Attribute]
  def typeListP[$: P]: P[Seq[Attribute]]
  def parenTypeListP[$: P]: P[Seq[Attribute]]
  def regionP[$: P](entryArgs: Seq[(String, Attribute)] = Seq.empty): P[Region]
  def operandP[$: P, A <: Attribute](name: String, typ: A): P[Value[A]]
  def resultP[$: P, A <: Attribute](name: String, typ: A): P[Result[A]]
  def valueIdAndTypeP[$: P]: P[(String, Attribute)]
  def attributeDictionaryP[$: P]: P[Map[String, Attribute]]
  def optionalAttributesP[$: P]: P[Map[String, Attribute]]
  def operationP[$: P]: P[Operation]
  def moduleP[$: P]: P[Operation]

  def generateOperationP[$: P](
      opName: String,
      resultsNames: Seq[String] = Seq.empty,
      operandsNames: Seq[String] = Seq.empty,
      successors: Seq[Block] = Seq.empty,
      properties: Map[String, Attribute] = Map(),
      regions: Seq[Region] = Seq.empty,
      attributes: Map[String, Attribute] = Map(),
      resultsTypes: Seq[Attribute] = Seq.empty,
      operandsTypes: Seq[Attribute] = Seq.empty,
  ): P[Operation]

  def parse[T](
      input: ParserInputSource,
      parser: P[?] => P[T] = moduleP(using _),
      verboseFailures: Boolean = false,
      startIndex: Int = 0,
      instrument: fastparse.internal.Instrument = null,
  ): Parsed[T]

  def error(failure: Failure, lineOffset: Int = 0): String

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||   TOP-LEVEL ENTRY POINTS      ||
\*≡==---==≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡==---==≡*/

inline def attributeP[$: P](using p: Parser): P[Attribute] = p.attributeP
inline def typeP[$: P](using p: Parser): P[Attribute] = p.typeP
inline def typeListP[$: P](using p: Parser): P[Seq[Attribute]] = p.typeListP

inline def parenTypeListP[$: P](using p: Parser): P[Seq[Attribute]] =
  p.parenTypeListP

inline def regionP[$: P](
    entryArgs: Seq[(String, Attribute)] = Seq.empty
)(using p: Parser): P[Region] = p.regionP(entryArgs)

inline def operandP[$: P, A <: Attribute](name: String, typ: A)(using
    p: Parser
): P[Value[A]] = p.operandP(name, typ)

inline def resultP[$: P, A <: Attribute](name: String, typ: A)(using
    p: Parser
): P[Result[A]] = p.resultP(name, typ)

inline def valueIdAndTypeP[$: P](using p: Parser): P[(String, Attribute)] =
  p.valueIdAndTypeP

inline def attributeDictionaryP[$: P](using
    p: Parser
): P[Map[String, Attribute]] = p.attributeDictionaryP

inline def optionalAttributesP[$: P](using
    p: Parser
): P[Map[String, Attribute]] = p.optionalAttributesP

def operationP[$: P](using p: Parser): P[Operation] = p.operationP
def moduleP[$: P](using p: Parser): P[Operation] = p.moduleP

inline def attrOfOrP[A <: Attribute](default: A)(using
    Parser
)(using P[Any]) =
  attributeP.orElse(default).flatMap(_ match
    case attr: A => Pass(attr)
    case _       => Fail("Expected sumin, got sumin else"))

inline def attrOfP[A <: Attribute](using
    Parser
)(using P[Any]) =
  attributeP.flatMap(_ match
    case attr: A => Pass(attr)
    case _       => Fail("Expected sumin, got sumin else"))

inline def typeOfOrP[T <: TypeAttribute](default: T)(using
    Parser
)(using P[Any]) =
  typeP.orElse(default).flatMap(_ match
    case tpe: T => Pass(tpe)
    case _      => Fail("Expected sumin, got sumin else"))

inline def typeOfP[T <: TypeAttribute](using
    Parser
)(using P[Any]) =
  typeP.flatMap(_ match
    case tpe: T => Pass(tpe)
    case _      => Fail("Expected sumin, got sumin else"))

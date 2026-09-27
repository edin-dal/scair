package scair.clair

import fastparse.P
import scair.clair.macros.deriveAttrDefs
import scair.ir.*
import scair.parse.Parser

import scala.quoted.*

trait AttributeCustomParser[T <: Attribute]:
  export scair.parse.whitespace

  def parse[$: P](using
      Parser
  ): P[T]

trait AttrDefs[T <: Attribute] extends AttributeCompanion[T]:
  def parameters(attr: T): Seq[Attribute]

  override def parse[$: P](using Parser): P[T]

object AttrDefs:

  inline def derived[T <: Attribute]: AttrDefs[T] = ${
    deriveAttrDefs[T]
  }

def summonAttributeCompanionsMacroRec[T <: Tuple: Type](using
    Quotes
): Seq[Expr[AttributeCompanion[?]]] =
  import quotes.reflect.*
  Type.of[T] match
    case '[type a <: Attribute; `a` *: ts] =>
      val dat = Expr.summon[AttributeCompanion[a]]
        .getOrElse(
          report
            .errorAndAbort(
              f"Could not summon AttributeCompanion for ${Type.show[a]}"
            )
        )
      dat +: summonAttributeCompanionsMacroRec[ts]
    case '[EmptyTuple] => Seq()

def summonAttributeCompanionsMacro[T <: Tuple: Type](using
    Quotes
): Expr[Seq[AttributeCompanion[?]]] =
  Expr.ofSeq(summonAttributeCompanionsMacroRec[T])

inline def summonAttributeCompanions[T <: Tuple]: Seq[AttributeCompanion[?]] =
  ${ summonAttributeCompanionsMacro[T] }

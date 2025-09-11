package scair.clair.macros

import fastparse.ParsingRun
import scair.AttrParser
import scair.Printer
import scair.ir.*

import scala.quoted.*
import scala.util.Failure
import scala.util.Success
import scala.util.Try

trait DerivedAttributeCompanion[T <: Attribute] extends AttributeCompanion {
  def parameters(attr: T): Seq[Attribute | Seq[Attribute]]
  override def parse[$: ParsingRun](p: AttrParser): ParsingRun[T]
}

object DerivedAttributeCompanion {

  inline def derived[T <: Attribute]: DerivedAttributeCompanion[T] = ${
    derivedAttributeCompanion[T]
  }

}

trait DerivedOperationCompanion[T] extends OperationCompanion {

  companion =>

  def operands(adtOp: T): Seq[Value[Attribute]]
  def successors(adtOp: T): Seq[Block]
  def results(adtOp: T): Seq[Result[Attribute]]
  def regions(adtOp: T): Seq[Region]
  def properties(adtOp: T): Map[String, Attribute]
  def custom_print(adtOp: T, p: Printer)(using indentLevel: Int): Unit
  def constraint_verify(adtOp: T): Either[String, Operation]

  case class UnstructuredOp(
      override val operands: Seq[Value[Attribute]] = Seq(),
      override val successors: Seq[Block] = Seq(),
      override val results: Seq[Result[Attribute]] = Seq(),
      override val regions: Seq[Region] = Seq(),
      override val properties: Map[String, Attribute] =
        Map.empty[String, Attribute],
      override val attributes: DictType[String, Attribute] =
        DictType.empty[String, Attribute]
  ) extends BaseOperation(
        name =
          name, // DEFINED IN OperationCompanion, derived in DerivedOperationCompanion Companion
        operands,
        successors,
        results,
        regions,
        properties,
        attributes
      ) {

    override def structured = Try(companion.structure(this)) match {
      case Failure(e)  => Left(e.toString())
      case Success(op) => op.asInstanceOf[Operation].structured
    }

    override def verify(): Either[String, Operation] = {
      structured.flatMap(op => op.verify())
    }

  }

  def apply(
      operands: Seq[Value[Attribute]] = Seq(),
      successors: Seq[scair.ir.Block] = Seq(),
      results: Seq[Result[Attribute]] = Seq(),
      regions: Seq[Region] = Seq(),
      properties: Map[String, Attribute] = Map.empty[String, Attribute],
      attributes: DictType[String, Attribute] =
        DictType.empty[String, Attribute]
  ): UnstructuredOp | T & Operation

  def destructure(adtOp: T): UnstructuredOp
  def structure(unstrucOp: UnstructuredOp): T

}

object DerivedOperationCompanion {

  inline def derived[T <: Operation]: DerivedOperationCompanion[T] = ${
    deriveOperationCompanion[T]
  }

}

def summonMLIRTraitsMacroRec[T <: Tuple: Type](using
    Quotes
): Seq[Expr[DerivedOperationCompanion[?]]] =
  import quotes.reflect.*
  Type.of[T] match
    case '[t *: ts] =>
      val dat = Expr
        .summon[DerivedOperationCompanion[t]]
        // TODO: Come on.
        .getOrElse(
          report.errorAndAbort(
            "summonDialect's operation type parameters should be for derived operations; Please use the Dialect constructor otherwise."
          )
        )
      dat +: summonMLIRTraitsMacroRec[ts]

    case '[EmptyTuple] => Seq()

def summonMLIRTraitsMacro[T <: Tuple: Type](using
    Quotes
): Expr[Seq[DerivedOperationCompanion[?]]] =
  Expr.ofSeq(summonMLIRTraitsMacroRec[T])

def summonAttributeTraitsMacroRec[T <: Tuple: Type](using
    Quotes
): Seq[Expr[DerivedAttributeCompanion[?]]] =
  import quotes.reflect.*
  Type.of[T] match
    case '[type t <: Attribute; `t` *: ts] =>
      val dat = Expr
        .summon[DerivedAttributeCompanion[t]]
        .getOrElse(
          report.errorAndAbort(
            "summonDialect's attribute type parameters should be for derived attributes; Please use the function arguments otherwise."
          )
        )
      dat +: summonAttributeTraitsMacroRec[ts]
    case '[EmptyTuple] => Seq()

def summonAttributeTraitsMacro[T <: Tuple: Type](using
    Quotes
): Expr[Seq[DerivedAttributeCompanion[?]]] =
  Expr.ofSeq(summonAttributeTraitsMacroRec[T])

inline def summonAttributeTraits[T <: Tuple]
    : Seq[DerivedAttributeCompanion[?]] =
  ${ summonAttributeTraitsMacro[T] }

inline def summonMLIRTraits[T <: Tuple]: Seq[DerivedOperationCompanion[?]] =
  ${ summonMLIRTraitsMacro[T] }

inline def summonDialect[Attributes <: Tuple, Operations <: Tuple](
    attributes: Seq[AttributeCompanion]
): Dialect =
  new Dialect(
    summonMLIRTraits[Operations],
    attributes ++ summonAttributeTraits[Attributes]
  )

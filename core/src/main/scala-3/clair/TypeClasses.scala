package scair.clair.macros

import fastparse.ParsingRun
import scair.AttrParser
import scair.Printer
import scair.ir.*

import scala.quoted.*
import scala.util.Failure
import scala.util.Success
import scala.util.Try

// ████████╗ ██╗░░░██╗ ██████╗░ ███████╗
// ╚══██╔══╝ ╚██╗░██╔╝ ██╔══██╗ ██╔════╝
// ░░░██║░░░ ░╚████╔╝░ ██████╔╝ █████╗░░
// ░░░██║░░░ ░░╚██╔╝░░ ██╔═══╝░ ██╔══╝░░
// ░░░██║░░░ ░░░██║░░░ ██║░░░░░ ███████╗
// ░░░╚═╝░░░ ░░░╚═╝░░░ ╚═╝░░░░░ ╚══════╝
//
// ░█████╗░ ██╗░░░░░ ░█████╗░ ░██████╗ ░██████╗ ███████╗ ░██████╗
// ██╔══██╗ ██║░░░░░ ██╔══██╗ ██╔════╝ ██╔════╝ ██╔════╝ ██╔════╝
// ██║░░╚═╝ ██║░░░░░ ███████║ ╚█████╗░ ╚█████╗░ █████╗░░ ╚█████╗░
// ██║░░██╗ ██║░░░░░ ██╔══██║ ░╚═══██╗ ░╚═══██╗ ██╔══╝░░ ░╚═══██╗
// ╚█████╔╝ ███████╗ ██║░░██║ ██████╔╝ ██████╔╝ ███████╗ ██████╔╝
// ░╚════╝░ ╚══════╝ ╚═╝░░╚═╝ ╚═════╝░ ╚═════╝░ ╚══════╝ ╚═════╝░

trait DerivedAttributeCompanion[T <: Attribute] extends AttributeCompanion[T]:
  def parameters(attr: T): Seq[Attribute | Seq[Attribute]]
  override def parse[$: ParsingRun](p: AttrParser): ParsingRun[T]

object DerivedAttributeCompanion:

  inline def derived[T <: Attribute]: DerivedAttributeCompanion[T] = ${
    derivedAttributeCompanion[T]
  }

trait DerivedOperationCompanion[T <: Operation] extends OperationCompanion[T]:

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
  ) extends Operation:

    override def updated(
        operands: Seq[Value[Attribute]] = operands,
        successors: Seq[Block] = successors,
        results: Seq[Result[Attribute]] = results.map(_.typ).map(Result(_)),
        regions: Seq[Region] = detached_regions,
        properties: Map[String, Attribute] = properties,
        attributes: DictType[String, Attribute] = attributes
    ): Operation =
      UnstructuredOp(
        operands,
        successors,
        results,
        regions,
        properties,
        attributes
      )

    override def structured = Try(companion.structure(this)) match
      case Failure(e)  => Left(e.toString())
      case Success(op) => op.asInstanceOf[Operation].structured

    override def verify(): Either[String, Operation] =
      structured.flatMap(op => op.verify())

    override def name = companion.name

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

object DerivedOperationCompanion:

  inline def derived[T <: Operation]: DerivedOperationCompanion[T] = ${
    deriveOperationCompanion[T]
  }

def summonOperationCompanionsMacroRec[T <: Tuple: Type](using
    Quotes
): Seq[Expr[OperationCompanion[?]]] =
  import quotes.reflect.*
  Type.of[T] match
    case '[type o <: Operation; o *: ts] =>
      val dat = Expr
        .summon[OperationCompanion[o]]
        .getOrElse(
          report.errorAndAbort(
            f"Could not summon OperationCompanion for ${Type.show[o]}"
          )
        )
      dat +: summonOperationCompanionsMacroRec[ts]

    case '[EmptyTuple] => Seq()

def summonOperationCompanionsMacro[T <: Tuple: Type](using
    Quotes
): Expr[Seq[OperationCompanion[?]]] =
  Expr.ofSeq(summonOperationCompanionsMacroRec[T])

def summonAttributeCompanionsMacroRec[T <: Tuple: Type](using
    Quotes
): Seq[Expr[AttributeCompanion[?]]] =
  import quotes.reflect.*
  Type.of[T] match
    case '[type a <: Attribute; `a` *: ts] =>
      val dat = Expr
        .summon[AttributeCompanion[a]]
        .getOrElse(
          report.errorAndAbort(
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

inline def summonOperationCompanions[T <: Tuple]: Seq[OperationCompanion[?]] =
  ${ summonOperationCompanionsMacro[T] }

inline def summonDialect[Attributes <: Tuple, Operations <: Tuple]: Dialect =
  Dialect(
    summonOperationCompanions[Operations],
    summonAttributeCompanions[Attributes]
  )

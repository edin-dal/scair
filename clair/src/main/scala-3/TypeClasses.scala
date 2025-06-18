package scair.clair.macros

import scair.ir.*
import scala.compiletime.*
import scala.util.{Try, Success, Failure}

trait DerivedAttributeCompanion[T] extends AttributeCompanionI[T] {
  def parameters(attr: T): Seq[Attribute | Seq[Attribute]]
  extension (op: T) override def AttributeTrait = this
}

trait DerivedAttribute[name <: String, T] extends ParametrizedAttribute {

  this: T =>

  given companion: DerivedAttributeCompanion[T] = deferred
  override val name: String = companion.name

  override val parameters: Seq[Attribute | Seq[Attribute]] =
    companion.parameters(this)

}

object DerivedAttributeCompanion {

  inline def derived[T]: DerivedAttributeCompanion[T] = ${ derivedAttributeCompanion[T] }

}

trait DerivedOperation[name <: String, T] extends Operation {

  this: T =>

  given companion: DerivedOperationCompanion[T] = deferred

  override def updated(
      operands: Seq[Value[Attribute]],
      successors: Seq[Block],
      results: Seq[Result[Attribute]],
      regions: Seq[Region],
      properties: Map[String, Attribute],
      attributes: DictType[String, Attribute]
  ) =
    companion(
      operands = operands,
      successors = successors,
      results = results,
      regions = regions,
      properties = properties,
      attributes = attributes
    )

  def name: String = companion.name
  // TODO: refactor this to have efficient generic accessors here and combine that in unverify instead.
  def operands: Seq[Value[Attribute]] = companion.operands(this)
  def successors: Seq[Block] = companion.successors(this)
  def results: Seq[Result[Attribute]] = companion.results(this)
  def regions: Seq[Region] = companion.regions(this)
  def properties: Map[String, Attribute] = companion.properties(this)

}


trait DerivedOperationCompanion[T] extends OperationCompanion {

  companion =>

  def operands(adtOp: T): Seq[Value[Attribute]]
  def successors(adtOp: T): Seq[Block]
  def results(adtOp: T): Seq[Result[Attribute]]
  def regions(adtOp: T): Seq[Region]
  def properties(adtOp: T): Map[String, Attribute]

  case class UnverifiedOp(
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

    override def copy(
        operands: Seq[Value[Attribute]],
        successors: Seq[Block],
        results: Seq[Result[Attribute]],
        regions: Seq[Region],
        properties: Map[String, Attribute],
        attributes: DictType[String, Attribute]
    ) = {
      UnregisteredOperation(
        name = name,
        operands = operands,
        successors = successors,
        results = results,
        regions = regions,
        properties = properties,
        attributes = attributes
      )
    }

    override def verify(): Either[String, Operation] = {
      Try(companion.verify(this)) match {
        case Success(op) =>
          op match {
            case adtOp: DerivedOperation[_, ?] =>
              adtOp.verify()
            case _ =>
              Left("Internal Error: Operation is not a DerivedOperation")
          }
        case Failure(e) => Left(e.toString())
      }
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
  ): UnverifiedOp

  def unverify(adtOp: T): UnverifiedOp
  def verify(unverOp: UnverifiedOp): T

}

trait MLIRName[name <: String]


object DerivedOperationCompanion {

  inline def derived[T <: Operation]: DerivedOperationCompanion[T] = ${
    deriveOperationCompanion[T]
  }

  
}

inline def summonAttributeTraits[T <: Tuple]
    : Seq[DerivedAttributeCompanion[?]] =
  inline erasedValue[T] match
    case _: (t *: ts) =>
      summonInline[DerivedAttributeCompanion[t]] +: summonAttributeTraits[ts]
    case _: EmptyTuple => Seq()

inline def summonMLIRTraits[T <: Tuple]: Seq[DerivedOperationCompanion[?]] =
  inline erasedValue[T] match
    case _: (t *: ts) =>
      summonInline[DerivedOperationCompanion[t]] +: summonMLIRTraits[ts]
    case _: EmptyTuple => Seq()

inline def summonDialect[Attributes <: Tuple, Operations <: Tuple](
    attributes: Seq[AttributeCompanion]
): Dialect =
  new Dialect(
    summonMLIRTraits[Operations],
    attributes ++ summonAttributeTraits[Attributes]
  )
// RUN: scala full-classpath %s - | filecheck %s

import scair.scairdl.constraints._
import scair.clair.mirrored._
import scair.dialects.builtin.IntegerAttr
import scair.scairdl.irdef._
import scair.ir.{DataAttribute, AttributeObject}

object SampleData extends AttributeObject {
  override def name: String = "sample"
}

case class SampleData(val d: String) extends DataAttribute[String]("sample", d)

case class Complex(
    e1: Operand[IntegerAttr]
) extends AttributeFE

case class Norm(
    e1: Operand[IntegerAttr],
    e2: Result[AnyAttribute],
    e3: Region
) extends OperationFE

case class Mul(
    e1: Operand[IntegerAttr],
    e2: Result[AnyAttribute]
) extends OperationFE

object CMathGen
    extends ScaIRDLDialect(
      summonDialect[(Complex, Norm, Mul)](
        "CMath",
        Seq(),
        Seq(new AttrEscapeHatch[SampleData])
      )
    )

// CHECK:       package scair.dialects.cmath

// CHECK:       import scair.ir._
// CHECK-NEXT:  import scair.Parser
// CHECK-NEXT:  import scair.Parser.whitespace

// CHECK:       import scair.dialects.builtin._
// CHECK-NEXT:  import scair.scairdl.constraints._
// CHECK-NEXT:  import scair.scairdl.constraints.attr2constraint

// CHECK:       import SampleData

// CHECK:       object Norm extends MLIROperationObject {
// CHECK-NEXT:    override def name = "cmath.norm"
// CHECK-NEXT:    override def factory = Norm.apply

// CHECK:       }

// CHECK:       case class Norm(
// CHECK-NEXT:      override val operands: ListType[Value[Attribute]] = ListType(),
// CHECK-NEXT:      override val successors: ListType[Block] = ListType(),
// CHECK-NEXT:      results_types: ListType[Attribute] = ListType(),
// CHECK-NEXT:      override val regions: ListType[Region] = ListType(),
// CHECK-NEXT:      override val dictionaryProperties: DictType[String, Attribute] =
// CHECK-NEXT:        DictType.empty[String, Attribute],
// CHECK-NEXT:      override val dictionaryAttributes: DictType[String, Attribute] =
// CHECK-NEXT:        DictType.empty[String, Attribute]
// CHECK-NEXT:  ) extends RegisteredOperation(name = "cmath.norm", operands, successors, results_types, regions, dictionaryProperties, dictionaryAttributes) {

// CHECK:         def e1: Value[Attribute] = operands(0)
// CHECK-NEXT:    def e1_=(new_operand: Value[Attribute]): Unit = {operands(0) = new_operand}

// CHECK:         def e2: Result[Attribute] = results(0)

// CHECK:         def e3: Region = regions(0)
// CHECK-NEXT:    def e3_=(new_region: Region): Unit = {regions(0) = new_region}

// CHECK:         val e1_constr = BaseAttr[scair.dialects.builtin.IntegerAttr]()
// CHECK-NEXT:    val e2_constr = AnyAttr

// CHECK:         override def custom_verify(): Unit =
// CHECK-NEXT:      val verification_context = new ConstraintContext()

// CHECK:           if (operands.length != 1) then throw new Exception(s"Expected 1 operands, got ${operands.length}")
// CHECK-NEXT:      if (results.length != 1) then throw new Exception(s"Expected 1 results, got ${results.length}")
// CHECK-NEXT:      if (regions.length != 1) then throw new Exception(s"Expected 1 regions, got ${regions.length}")
// CHECK-NEXT:      if (successors.length != 0) then throw new Exception(s"Expected 0 successors, got ${successors.length}")

// CHECK:           e1_constr.verify(e1.typ, verification_context)
// CHECK-NEXT:      e2_constr.verify(e2.typ, verification_context)

// CHECK:       }

// CHECK:       object Mul extends MLIROperationObject {
// CHECK-NEXT:    override def name = "cmath.mul"
// CHECK-NEXT:    override def factory = Mul.apply

// CHECK:       }

// CHECK:       case class Mul(
// CHECK-NEXT:      override val operands: ListType[Value[Attribute]] = ListType(),
// CHECK-NEXT:      override val successors: ListType[Block] = ListType(),
// CHECK-NEXT:      results_types: ListType[Attribute] = ListType(),
// CHECK-NEXT:      override val regions: ListType[Region] = ListType(),
// CHECK-NEXT:      override val dictionaryProperties: DictType[String, Attribute] =
// CHECK-NEXT:        DictType.empty[String, Attribute],
// CHECK-NEXT:      override val dictionaryAttributes: DictType[String, Attribute] =
// CHECK-NEXT:        DictType.empty[String, Attribute]
// CHECK-NEXT:  ) extends RegisteredOperation(name = "cmath.mul", operands, successors, results_types, regions, dictionaryProperties, dictionaryAttributes) {

// CHECK:         def e1: Value[Attribute] = operands(0)
// CHECK-NEXT:    def e1_=(new_operand: Value[Attribute]): Unit = {operands(0) = new_operand}

// CHECK:         def e2: Result[Attribute] = results(0)

// CHECK:         val e1_constr = BaseAttr[scair.dialects.builtin.IntegerAttr]()
// CHECK-NEXT:    val e2_constr = AnyAttr

// CHECK:         override def custom_verify(): Unit =
// CHECK-NEXT:      val verification_context = new ConstraintContext()

// CHECK:           if (operands.length != 1) then throw new Exception(s"Expected 1 operands, got ${operands.length}")
// CHECK-NEXT:      if (results.length != 1) then throw new Exception(s"Expected 1 results, got ${results.length}")
// CHECK-NEXT:      if (regions.length != 0) then throw new Exception(s"Expected 0 regions, got ${regions.length}")
// CHECK-NEXT:      if (successors.length != 0) then throw new Exception(s"Expected 0 successors, got ${successors.length}")

// CHECK:           e1_constr.verify(e1.typ, verification_context)
// CHECK-NEXT:      e2_constr.verify(e2.typ, verification_context)

// CHECK:       }

// CHECK:       object Complex extends AttributeObject {
// CHECK-NEXT:    override def name = "cmath.complex"
// CHECK-NEXT:    override def factory = Complex.apply
// CHECK-NEXT:  }

// CHECK:       case class Complex(override val parameters: Seq[Attribute]) extends ParametrizedAttribute(name = "cmath.complex", parameters = parameters)  {
// CHECK-NEXT:    override def custom_verify(): Unit =
// CHECK-NEXT:      if (parameters.length != 1) then throw new Exception(s"Expected 1 parameters, got ${parameters.length}")
// CHECK-NEXT:  }

// CHECK:       val CMathDialect: Dialect = new Dialect(
// CHECK-NEXT:    operations = Seq(Norm, Mul),
// CHECK-NEXT:    attributes = Seq(Complex, SampleData)
// CHECK-NEXT:  )

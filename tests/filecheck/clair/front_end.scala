// RUN: scala full-classpath %s - | filecheck %s

import scair.scairdl.constraints._
import scair.clair.mirrored._
import scair.dialects.builtin.IntegerAttr
import scair.scairdl.irdef._

enum CMathAttrs extends DialectAttribute:

  case Complex(
      e1: Operand[IntegerAttr]
  )

enum CMathOps extends DialectOperation:

  case Norm(
      e1: Operand[IntegerAttr],
      e2: Result[AnyAttribute],
      e3: Region
  )
  case Mul[Operation](
      e1: Operand[IntegerAttr],
      e2: Result[AnyAttribute]
  )

object CMath extends ScaIRDLDialect(summonDialect[CMathOps, CMathAttrs])

// CHECK:       import scair.ir._
// CHECK-NEXT:  import scair.dialects.builtin._
// CHECK-NEXT:  import scair.scairdl.constraints._
// CHECK-NEXT:  import scair.scairdl.constraints.attr2constraint

// CHECK:       object Norm extends DialectOperation {
// CHECK-NEXT:    override def name = "cmath.norm"
// CHECK-NEXT:    override def factory = Norm.apply
// CHECK-NEXT:  }

// CHECK:       case class Norm(
// CHECK-NEXT:      override val operands: ListType[Value[Attribute]] = ListType(),
// CHECK-NEXT:      override val successors: ListType[Block] = ListType(),
// CHECK-NEXT:      override val results: ListType[Value[Attribute]] = ListType(),
// CHECK-NEXT:      override val regions: ListType[Region] = ListType(),
// CHECK-NEXT:      override val dictionaryProperties: DictType[String, Attribute] =
// CHECK-NEXT:        DictType.empty[String, Attribute],
// CHECK-NEXT:      override val dictionaryAttributes: DictType[String, Attribute] =
// CHECK-NEXT:        DictType.empty[String, Attribute]
// CHECK-NEXT:  ) extends RegisteredOperation(name = "cmath.norm") {

// CHECK:         def e1: Value[Attribute] = operands(0)
// CHECK-NEXT:    def e1_=(new_operand: Value[Attribute]): Unit = {operands(0) = new_operand}

// CHECK:         def e2: Value[Attribute] = results(0)
// CHECK-NEXT:    def e2_=(new_result: Value[Attribute]): Unit = {results(0) = new_result}

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
// CHECK-NEXT:      if (dictionaryProperties.size != 0) then throw new Exception("Expected 0 properties, got dictionaryProperties.size")
// CHECK-NEXT:      if (dictionaryAttributes.size != 0) then throw new Exception("Expected 0 attributes, got dictionaryAttributes.size")

// CHECK:           e1_constr.verify(e1.typ, verification_context)
// CHECK-NEXT:      e2_constr.verify(e2.typ, verification_context)

// CHECK:       }

// CHECK:       object Mul extends DialectOperation {
// CHECK-NEXT:    override def name = "cmath.mul"
// CHECK-NEXT:    override def factory = Mul.apply
// CHECK-NEXT:  }

// CHECK:       case class Mul(
// CHECK-NEXT:      override val operands: ListType[Value[Attribute]] = ListType(),
// CHECK-NEXT:      override val successors: ListType[Block] = ListType(),
// CHECK-NEXT:      override val results: ListType[Value[Attribute]] = ListType(),
// CHECK-NEXT:      override val regions: ListType[Region] = ListType(),
// CHECK-NEXT:      override val dictionaryProperties: DictType[String, Attribute] =
// CHECK-NEXT:        DictType.empty[String, Attribute],
// CHECK-NEXT:      override val dictionaryAttributes: DictType[String, Attribute] =
// CHECK-NEXT:        DictType.empty[String, Attribute]
// CHECK-NEXT:  ) extends RegisteredOperation(name = "cmath.mul") {

// CHECK:         def e1: Value[Attribute] = operands(0)
// CHECK-NEXT:    def e1_=(new_operand: Value[Attribute]): Unit = {operands(0) = new_operand}

// CHECK:         def e2: Value[Attribute] = results(0)
// CHECK-NEXT:    def e2_=(new_result: Value[Attribute]): Unit = {results(0) = new_result}

// CHECK:         val e1_constr = BaseAttr[scair.dialects.builtin.IntegerAttr]()
// CHECK-NEXT:    val e2_constr = AnyAttr

// CHECK:         override def custom_verify(): Unit =
// CHECK-NEXT:      val verification_context = new ConstraintContext()

// CHECK:           if (operands.length != 1) then throw new Exception(s"Expected 1 operands, got ${operands.length}")
// CHECK-NEXT:      if (results.length != 1) then throw new Exception(s"Expected 1 results, got ${results.length}")
// CHECK-NEXT:      if (regions.length != 0) then throw new Exception(s"Expected 0 regions, got ${regions.length}")
// CHECK-NEXT:      if (successors.length != 0) then throw new Exception(s"Expected 0 successors, got ${successors.length}")
// CHECK-NEXT:      if (dictionaryProperties.size != 0) then throw new Exception("Expected 0 properties, got dictionaryProperties.size")
// CHECK-NEXT:      if (dictionaryAttributes.size != 0) then throw new Exception("Expected 0 attributes, got dictionaryAttributes.size")

// CHECK:           e1_constr.verify(e1.typ, verification_context)
// CHECK-NEXT:      e2_constr.verify(e2.typ, verification_context)

// CHECK:       }

// CHECK:       object Complex extends DialectAttribute {
// CHECK-NEXT:    override def name = "cmath.complex"
// CHECK-NEXT:    override def factory = Complex.apply
// CHECK-NEXT:  }

// CHECK:       case class Complex(override val parameters: Seq[Attribute]) extends ParametrizedAttribute(name = "cmath.complex", parameters = parameters)  {
// CHECK-NEXT:    override def custom_verify(): Unit =
// CHECK-NEXT:      if (parameters.length != 1) then throw new Exception("Expected 1 parameters, got parameters.length")
// CHECK-NEXT:  }

// CHECK:       val CMath: Dialect = new Dialect(
// CHECK-NEXT:    operations = Seq(Norm, Mul),
// CHECK-NEXT:    attributes = Seq(Complex)
// CHECK-NEXT:  )

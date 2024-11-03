// RUN: scala -classpath ../../target/scala-3.3.1/classes/ %s | filecheck %s

import scair.clair.ir._
import scair.dialects.builtin.IntData
import scair.scairdl.constraints._

object Main {
  def main(args: Array[String]) = {
    val dialect = DialectDef(
      "dialect",
      ListType(
        OperationDef(
          "dialect.name2",
          "NameOp",
          operands = List(
            OperandDef("map", BaseAttr[IntData]()),
            OperandDef("map2", EqualAttr(IntData(5))),
            OperandDef(
              "map3",
              AnyOf(Seq(EqualAttr(IntData(5)), EqualAttr(IntData(6))))
            )
          ),
          regions = List(RegionDef("testregion")),
          successors = List(SuccessorDef("testsuccessor"))
        )
      ),
      ListType(AttributeDef("dialect.name1", "NameAttr", typee = 1))
    )

    println(dialect.print(0))
  }
}

// CHECK: import scair.ir._
// CHECK: import scair.dialects.builtin._
// CHECK: import scair.scairdl.constraints._
// CHECK: object NameOp extends DialectOperation {
// CHECK:   override def name = "dialect.name2"
// CHECK:   override def factory = NameOp.apply
// CHECK: }
// CHECK: case class NameOp(
// CHECK:     override val operands: ListType[Value[Attribute]] = ListType(),
// CHECK:     override val successors: ListType[Block] = ListType(),
// CHECK:     override val results: ListType[Value[Attribute]] = ListType(),
// CHECK:     override val regions: ListType[Region] = ListType(),
// CHECK:     override val dictionaryProperties: DictType[String, Attribute] =
// CHECK:       DictType.empty[String, Attribute],
// CHECK:     override val dictionaryAttributes: DictType[String, Attribute] =
// CHECK:       DictType.empty[String, Attribute]
// CHECK: ) extends RegisteredOperation(name = "dialect.name2") {
// CHECK:   def map: Value[Attribute] = operands(0)
// CHECK:   def map_=(value: Value[Attribute]): Unit = {operands(0) = value}
// CHECK:   def map2: Value[Attribute] = operands(1)
// CHECK:   def map2_=(value: Value[Attribute]): Unit = {operands(1) = value}
// CHECK:   def map3: Value[Attribute] = operands(2)
// CHECK:   def map3_=(value: Value[Attribute]): Unit = {operands(2) = value}
// CHECK:   def testregion: Region = regions(0)
// CHECK:   def testregion_=(value: Region): Unit = {regions(0) = value}
// CHECK:   def testsuccessor: Block = successors(0)
// CHECK:   def testsuccessor_=(value: Block): Unit = {successors(0) = value}
// CHECK:   val NameOp_CTX = new ConstraintContext()
// CHECK:   val map_constr = BaseAttr[scair.dialects.builtin.IntData]()
// CHECK:   val map2_constr = EqualAttr(IntData(5))
// CHECK:   val map3_constr = AnyOf(List(EqualAttr(IntData(5)), EqualAttr(IntData(6))))
// CHECK:   override def custom_verify(): Unit =
// CHECK:     if (operands.length != 3) then throw new Exception("Expected 3 operands, got operands.length")
// CHECK:     if (results.length != 0) then throw new Exception("Expected 0 results, got results.length")
// CHECK:     if (regions.length != 1) then throw new Exception("Expected 1 regions, got regions.length")
// CHECK:     if (successors.length != 1) then throw new Exception("Expected 1 successors, got successors.length")
// CHECK:     if (dictionaryProperties.size != 0) then throw new Exception("Expected 0 properties, got dictionaryProperties.size")
// CHECK:     if (dictionaryAttributes.size != 0) then throw new Exception("Expected 0 attributes, got dictionaryAttributes.size")
// CHECK:     map_constr.verify(map.typ, NameOp_CTX)
// CHECK:     map2_constr.verify(map2.typ, NameOp_CTX)
// CHECK:     map3_constr.verify(map3.typ, NameOp_CTX)
// CHECK: }
// CHECK: object NameAttr extends DialectAttribute {
// CHECK:   override def name = "dialect.name1"
// CHECK:   override def factory = NameAttr.apply
// CHECK: }
// CHECK: case class NameAttr(override val parameters: Seq[Attribute]) extends ParametrizedAttribute(name = "dialect.name1", parameters = parameters) with TypeAttribute {
// CHECK:   override def custom_verify(): Unit =
// CHECK:     if (parameters.length != 0) then throw new Exception("Expected 0 parameters, got parameters.length")
// CHECK: }
// CHECK: val dialect: Dialect = new Dialect(
// CHECK:   operations = Seq(NameOp),
// CHECK:   attributes = Seq(NameAttr)
// CHECK: )

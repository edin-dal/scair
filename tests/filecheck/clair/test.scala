// RUN: scala -classpath ../../target/scala-3.3.1/classes/ --main-class Main %s | filecheck %s

import scair.clair.ir._

object Main {
  def main(args: Array[String]) = {
    val dialect = DialectDef(
      "dialect",
      ListType(
        OperationDef(
          "dialect.name2",
          "NameOp",
          operands = List(
            OperandDef("map", AnyOf(List(RegularType("dialect", "name2")))),
            OperandDef("map2", Equal(RegularType("dialect", "name2"))),
            OperandDef(
              "map3",
              AnyOf(
                List(
                  RegularType("dialect", "name2"),
                  RegularType("dialect", "name1")
                )
              )
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

// CHECK:       import scair.ir._
// CHECK:       object NameOp extends DialectOperation {
// CHECK-NEXT:    override def name = "dialect.name2"
// CHECK-NEXT:    override def factory = NameOp.apply
// CHECK-NEXT:  }
// CHECK:       case class NameOp(
// CHECK-NEXT:      override val operands: ListType[Value[Attribute]] = ListType(),
// CHECK-NEXT:      override val successors: ListType[Block] = ListType(),
// CHECK-NEXT:      override val results: ListType[Value[Attribute]] = ListType(),
// CHECK-NEXT:      override val regions: ListType[Region] = ListType(),
// CHECK-NEXT:      override val dictionaryProperties: DictType[String, Attribute] =
// CHECK-NEXT:        DictType.empty[String, Attribute],
// CHECK-NEXT:      override val dictionaryAttributes: DictType[String, Attribute] =
// CHECK-NEXT:        DictType.empty[String, Attribute]
// CHECK-NEXT:  ) extends RegisteredOperation(name = "dialect.name2") {
// CHECK:         override def custom_verify(): Unit =
// CHECK-NEXT:      if (operands.length != 3) then throw new Exception("Expected 3 operands, got operands.length")
// CHECK-NEXT:      if (results.length != 0) then throw new Exception("Expected 0 results, got results.length")
// CHECK-NEXT:      if (regions.length != 1) then throw new Exception("Expected 1 regions, got regions.length")
// CHECK-NEXT:      if (successors.length != 1) then throw new Exception("Expected 1 successors, got successors.length")
// CHECK-NEXT:      if (dictionaryProperties.size != 0) then throw new Exception("Expected 0 properties, got dictionaryProperties.size")
// CHECK-NEXT:      if (dictionaryAttributes.size != 0) then throw new Exception("Expected 0 attributes, got dictionaryAttributes.size")
// CHECK:         def map: Value[Attribute] = operands(0)
// CHECK-NEXT:    def map_=(value: Value[Attribute]): Unit = {operands(0) = value}
// CHECK:         def map2: Value[Attribute] = operands(1)
// CHECK-NEXT:    def map2_=(value: Value[Attribute]): Unit = {operands(1) = value}
// CHECK:         def map3: Value[Attribute] = operands(2)
// CHECK-NEXT:    def map3_=(value: Value[Attribute]): Unit = {operands(2) = value}
// CHECK:         def testregion: Region = regions(0)
// CHECK-NEXT:    def testregion_=(value: Region): Unit = {regions(0) = value}
// CHECK:         def testsuccessor: Block = successors(0)
// CHECK-NEXT:    def testsuccessor_=(value: Block): Unit = {successors(0) = value}
// CHECK:       }
// CHECK:       object NameAttr extends DialectAttribute {
// CHECK-NEXT:    override def name = "dialect.name1"
// CHECK-NEXT:    override def factory = NameAttr.apply
// CHECK-NEXT:  }
// CHECK:       case class NameAttr(override val parameters: Seq[Attribute]) extends ParametrizedAttribute(name = "dialect.name1", parameters = parameters) with TypeAttribute {
// CHECK-NEXT:    override def custom_verify(): Unit =
// CHECK-NEXT:      if (parameters.length != 0) then throw new Exception("Expected 0 parameters, got parameters.length")
// CHECK-NEXT:  }
// CHECK:       val dialect: Dialect = new Dialect(
// CHECK-NEXT:    operations = Seq(NameOp),
// CHECK-NEXT:    attributes = Seq(NameAttr)
// CHECK-NEXT:  )

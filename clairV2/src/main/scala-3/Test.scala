package scair.test

import scair.ir.*
import scair.dialects.builtin.*
import scair.clairV2.macros.*
import scala.quoted.*

case class MulV2(
    lhs: Operand[IntegerType],
    rhs: Operand[IntegerType],
    result: Result[IntegerType],
    randProp: Property[StringData]
) extends ADTOperation derives MLIRTrait

object Main {

  object TestCases {
    val e = UnverifiedOp[MulV2](
      name = "mulv2",
      operands = ListType(Value[Attribute](typ = IntegerType(IntData(5), Unsigned)), Value[IntegerType](typ = IntegerType(IntData(5), Unsigned))),
      results_types = ListType[Attribute](IntegerType(IntData(25), Unsigned)),
      dictionaryProperties = DictType(("randProp" -> StringData("what")))
    )
    val f = MulV2(
      lhs = Value(IntegerType(IntData(5), Unsigned)),
      rhs = Value(IntegerType(IntData(5), Unsigned)),
      result = Result(IntegerType(IntData(25), Unsigned)),
      randProp = Property(StringData("what"))
    )
  }

  def testMacroE()(using opT: MLIRTrait[MulV2]): Unit = {
    val op = TestCases.e
    val specialedOp = opT.specializeUnverifiedOp(op)
    println(specialedOp)
  }

  def testMacroF()(using opT: MLIRTrait[MulV2]): Unit = {
    val op = TestCases.f
    val generalizedOp = opT.generalizeADTOp(op)
    println(generalizedOp)
  }

  def main(args: Array[String]): Unit = {
    testMacroE() // works ish
    testMacroF() // works
  }
}
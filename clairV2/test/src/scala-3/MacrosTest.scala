import scair.ir.*
import scair.dialects.builtin.*
import scair.clairV2.macros.*
import scala.quoted.*
import scair.Printer
import org.scalatest.*
import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import scala.collection.mutable.LinkedHashMap

case class MulV2(
    lhs: Operand[IntegerType],
    rhs: Operand[IntegerType],
    result: Result[IntegerType],
    randProp: Property[StringData]
) extends ADTOperation
    derives MLIRTrait

case class MulFull(
    operand1: Operand[IntegerType],
    operand2: Operand[IntegerType],
    result1: Result[IntegerType],
    result2: Result[IntegerType],
    randProp1: Property[StringData],
    randProp2: Property[StringData],
    reg1: Region,
    reg2: Region,
    succ1: Successor,
    succ2: Successor
) extends ADTOperation
    derives MLIRTrait

class MacrosTest extends AnyFlatSpec with BeforeAndAfter {

  object TestCases {

    val unverOp = UnverifiedOp[MulV2](
      name = "mulv2",
      operands = ListType(
        Value[Attribute](typ = IntegerType(IntData(5), Unsigned)),
        Value[IntegerType](typ = IntegerType(IntData(5), Unsigned))
      ),
      results_types = ListType[Attribute](IntegerType(IntData(25), Unsigned)),
      dictionaryProperties = DictType(("randProp" -> StringData("what")))
    )

    val adtOp = MulV2(
      lhs = Value(IntegerType(IntData(5), Unsigned)),
      rhs = Value(IntegerType(IntData(5), Unsigned)),
      result = Result(IntegerType(IntData(25), Unsigned)),
      randProp = Property(StringData("what"))
    )

    val adtOpAllFields = MulFull(
      operand1 = Value(IntegerType(IntData(5), Unsigned)),
      operand2 = Value(IntegerType(IntData(5), Unsigned)),
      result1 = Result(IntegerType(IntData(25), Unsigned)),
      result2 = Result(IntegerType(IntData(25), Unsigned)),
      randProp1 = Property(StringData("what1")),
      randProp2 = Property(StringData("what2")),
      reg1 = Region(Seq()),
      reg2 = Region(Seq()),
      succ1 = scair.ir.Block(),
      succ2 = scair.ir.Block()
    )

  }

  "Unverified instantiation" should "Correctly instantiates the UniverifiedOp" in {
    val opT = summon[MLIRTrait[MulV2]]

    val unverOp = opT.constructUnverifiedOp(
      operands = ListType(
        Value[Attribute](typ = IntegerType(IntData(5), Unsigned)),
        Value[IntegerType](typ = IntegerType(IntData(5), Unsigned))
      ),
      results_types = ListType[Attribute](IntegerType(IntData(25), Unsigned)),
      dictionaryProperties = DictType(("randProp" -> StringData("what")))
    )

    unverOp.name should be("mulv2")
    unverOp.operands should matchPattern {
      case ListType(
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned))
          ) =>
    }
    unverOp.results should matchPattern {
      case ListType(Result(IntegerType(IntData(25), Unsigned))) =>
    }
    unverOp.dictionaryProperties("randProp") should matchPattern {
      case StringData("what") =>
    }
  }

  "Conversion to Unverified" should "Correctly translate from ADT operation to Univerified Operation" in {
    val opT = summon[MLIRTrait[MulV2]]

    val op = TestCases.adtOp
    val unverOp = opT.unverify(op)

    unverOp.name should be("mulv2")
    unverOp.operands should matchPattern {
      case ListType(
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned))
          ) =>
    }
    unverOp.results should matchPattern {
      case ListType(Result(IntegerType(IntData(25), Unsigned))) =>
    }
    unverOp.dictionaryProperties("randProp") should matchPattern {
      case StringData("what") =>
    }
  }

  "Conversion to ADTOp" should "Correctly translate from Unverified operation to ADT Operation" in {
    val opT = summon[MLIRTrait[MulV2]]

    val op = TestCases.unverOp
    val adtOp = opT.verify(op)

    adtOp.lhs should matchPattern {
      case Value(IntegerType(IntData(5), Unsigned)) =>
    }
    adtOp.rhs should matchPattern {
      case Value(IntegerType(IntData(5), Unsigned)) =>
    }
    adtOp.result should matchPattern {
      case Value(IntegerType(IntData(25), Unsigned)) =>
    }
    adtOp.randProp should matchPattern { case Property(StringData("what")) => }
  }

  "ADTOp test for correctly passing around the same instances" should "test all fields of a ADTOp" in {
    val opT = summon[MLIRTrait[MulFull]]

    val op = TestCases.adtOpAllFields

    val unverOp = opT.unverify(op)
    val adtOp = opT.verify(unverOp)

    adtOp.operand1 equals unverOp.operands(0) should be(true)
    adtOp.operand2 equals unverOp.operands(1) should be(true)
    adtOp.result1 equals unverOp.results(0) should be(true)
    adtOp.result2 equals unverOp.results(1) should be(true)
    adtOp.randProp1.typ equals unverOp.dictionaryProperties(
      "randProp1"
    ) should be(true)
    adtOp.randProp2.typ equals unverOp.dictionaryProperties(
      "randProp2"
    ) should be(true)
    adtOp.reg1 equals unverOp.regions(0) should be(true)
    adtOp.reg2 equals unverOp.regions(1) should be(true)
    adtOp.succ1 equals unverOp.successors(0) should be(true)
    adtOp.succ2 equals unverOp.successors(1) should be(true)
  }

}

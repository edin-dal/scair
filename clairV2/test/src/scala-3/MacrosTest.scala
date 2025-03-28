import scair.ir.*
import scair.dialects.builtin.*
import scair.clairV2.macros.*
import scala.quoted.*
import scair.Printer
import org.scalatest.*
import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import scala.collection.mutable.LinkedHashMap

// TODO: create a better constructor for DenseArrayAttr with just integers

case class Mul(
    lhs: Operand[IntegerType],
    rhs: Operand[IntegerType],
    result: Result[IntegerType],
    randProp: Property[StringData]
) extends MLIRName["cmath.mul"]
    derives MLIRTrait

case class MulSingleVariadic(
    lhs: Operand[IntegerType],
    rhs: Variadic[Operand[IntegerType]],
    result: Variadic[Result[IntegerType]],
    randProp: Property[StringData]
) extends MLIRName["cmath.mulsinglevariadic"]
    derives MLIRTrait

case class MulMultiVariadic(
    lhs: Operand[IntegerType],
    rhs: Variadic[Operand[IntegerType]],
    mhs: Variadic[Operand[IntegerType]],
    result: Variadic[Result[IntegerType]],
    result2: Result[IntegerType],
    result3: Variadic[Result[IntegerType]],
    operandSegmentSizes: Property[DenseArrayAttr],
    resultSegmentSizes: Property[DenseArrayAttr]
) extends MLIRName["cmath.mulmultivariadic"]
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
) extends MLIRName["cmath.mulfull"]
    derives MLIRTrait

class MacrosTest extends AnyFlatSpec with BeforeAndAfter {

  object TestCases {

    val unverOp = UnverifiedOp[Mul](
      name = "cmath.mul",
      operands = ListType(
        Value[Attribute](typ = IntegerType(IntData(5), Unsigned)),
        Value[IntegerType](typ = IntegerType(IntData(5), Unsigned))
      ),
      results_types = ListType[Attribute](IntegerType(IntData(25), Unsigned)),
      dictionaryProperties = DictType(("randProp" -> StringData("what")))
    )

    val adtMulOp = Mul(
      lhs = Value(IntegerType(IntData(5), Unsigned)),
      rhs = Value(IntegerType(IntData(5), Unsigned)),
      result = Result(IntegerType(IntData(25), Unsigned)),
      randProp = Property(StringData("what"))
    )

    val adtMulOpAllFields = MulFull(
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

    val unverMulSinVarOp = UnverifiedOp[MulSingleVariadic](
      name = "cmath.mulsinglevariadic",
      operands = ListType(
        Value[Attribute](typ = IntegerType(IntData(5), Unsigned)),
        Value[IntegerType](typ = IntegerType(IntData(5), Unsigned)),
        Value[IntegerType](typ = IntegerType(IntData(5), Unsigned))
      ),
      results_types = ListType[Attribute](
        IntegerType(IntData(25), Unsigned),
        IntegerType(IntData(25), Unsigned)
      ),
      dictionaryProperties = DictType(("randProp" -> StringData("what")))
    )

    val adtMulSinVarOp = MulSingleVariadic(
      lhs = Value(IntegerType(IntData(5), Unsigned)),
      rhs = Variadic(
        Value(IntegerType(IntData(5), Unsigned)),
        Value(IntegerType(IntData(5), Unsigned))
      ),
      result = Variadic(
        Result(IntegerType(IntData(25), Unsigned)),
        Result(IntegerType(IntData(25), Unsigned))
      ),
      randProp = Property(StringData("what"))
    )

    val unverMulMulVarOp = UnverifiedOp[MulMultiVariadic](
      name = "cmath.mulmultivariadic",
      operands = ListType(
        Value[Attribute](typ = IntegerType(IntData(5), Unsigned)),
        Value[IntegerType](typ = IntegerType(IntData(5), Unsigned)),
        Value[IntegerType](typ = IntegerType(IntData(5), Unsigned)),
        Value[IntegerType](typ = IntegerType(IntData(5), Unsigned)),
        Value[IntegerType](typ = IntegerType(IntData(5), Unsigned)),
        Value[IntegerType](typ = IntegerType(IntData(5), Unsigned))
      ),
      results_types = ListType[Attribute](
        IntegerType(IntData(25), Unsigned),
        IntegerType(IntData(25), Unsigned),
        IntegerType(IntData(25), Unsigned),
        IntegerType(IntData(25), Unsigned),
        IntegerType(IntData(25), Unsigned),
        IntegerType(IntData(25), Unsigned)
      ),
      dictionaryProperties = DictType(
        ("operandSegmentSizes" -> DenseArrayAttr(
          IntegerType(IntData(32), Signless),
          Seq[IntegerAttr](
            IntegerAttr(IntData(1), IntegerType(IntData(32), Signless)),
            IntegerAttr(IntData(3), IntegerType(IntData(32), Signless)),
            IntegerAttr(IntData(2), IntegerType(IntData(32), Signless))
          )
        )),
        ("resultSegmentSizes" -> DenseArrayAttr(
          IntegerType(IntData(32), Signless),
          Seq[IntegerAttr](
            IntegerAttr(IntData(3), IntegerType(IntData(32), Signless)),
            IntegerAttr(IntData(1), IntegerType(IntData(32), Signless)),
            IntegerAttr(IntData(2), IntegerType(IntData(32), Signless))
          )
        ))
      )
    )

    val adtMulMulVarOp = MulMultiVariadic(
      lhs = Value(IntegerType(IntData(5), Unsigned)),
      rhs = Variadic(
        Value(IntegerType(IntData(5), Unsigned)),
        Value(IntegerType(IntData(5), Unsigned)),
        Value(IntegerType(IntData(5), Unsigned))
      ),
      mhs = Variadic(
        Value(IntegerType(IntData(5), Unsigned)),
        Value(IntegerType(IntData(5), Unsigned))
      ),
      result = Variadic(
        Result(IntegerType(IntData(5), Unsigned)),
        Result(IntegerType(IntData(5), Unsigned)),
        Result(IntegerType(IntData(5), Unsigned))
      ),
      result2 = Result(IntegerType(IntData(5), Unsigned)),
      result3 = Variadic(
        Result(IntegerType(IntData(5), Unsigned)),
        Result(IntegerType(IntData(5), Unsigned))
      ),
      operandSegmentSizes = Property(
        DenseArrayAttr(
          IntegerType(IntData(32), Signless),
          Seq[IntegerAttr](
            IntegerAttr(IntData(1), IntegerType(IntData(32), Signless)),
            IntegerAttr(IntData(3), IntegerType(IntData(32), Signless)),
            IntegerAttr(IntData(2), IntegerType(IntData(32), Signless))
          )
        )
      ),
      resultSegmentSizes = Property(
        DenseArrayAttr(
          IntegerType(IntData(32), Signless),
          Seq[IntegerAttr](
            IntegerAttr(IntData(1), IntegerType(IntData(32), Signless)),
            IntegerAttr(IntData(3), IntegerType(IntData(32), Signless)),
            IntegerAttr(IntData(2), IntegerType(IntData(32), Signless))
          )
        )
      )
    )

  }

  "Unverified instantiation" should "Correctly instantiates the UniverifiedOp" in {
    val opT = MLIRTrait.derived[Mul]

    val unverMulOp = opT(
      operands = ListType(
        Value[Attribute](typ = IntegerType(IntData(5), Unsigned)),
        Value[IntegerType](typ = IntegerType(IntData(5), Unsigned))
      ),
      results_types = ListType[Attribute](IntegerType(IntData(25), Unsigned)),
      dictionaryProperties = DictType(("randProp" -> StringData("what")))
    )

    unverMulOp.name should be("cmath.mul")
    unverMulOp.operands should matchPattern {
      case ListType(
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned))
          ) =>
    }
    unverMulOp.results should matchPattern {
      case ListType(Result(IntegerType(IntData(25), Unsigned))) =>
    }
    unverMulOp.dictionaryProperties("randProp") should matchPattern {
      case StringData("what") =>
    }
  }

  "Conversion to Unverified" should "Correctly translate from ADT operation to Univerified Operation" in {
    val opT = summon[MLIRTrait[Mul]]

    val op = TestCases.adtMulOp
    val unverMulOp = opT.unverify(op)

    unverMulOp.name should be("cmath.mul")
    unverMulOp.operands should matchPattern {
      case ListType(
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned))
          ) =>
    }
    unverMulOp.results should matchPattern {
      case ListType(Result(IntegerType(IntData(25), Unsigned))) =>
    }
    unverMulOp.dictionaryProperties("randProp") should matchPattern {
      case StringData("what") =>
    }
  }

  "Conversion to ADTOp" should "Correctly translate from Unverified operation to ADT Operation" in {
    val opT = summon[MLIRTrait[Mul]]

    val op = TestCases.unverOp
    val adtMulOp = opT.verify(op)

    adtMulOp.lhs should matchPattern {
      case Value(IntegerType(IntData(5), Unsigned)) =>
    }
    adtMulOp.rhs should matchPattern {
      case Value(IntegerType(IntData(5), Unsigned)) =>
    }
    adtMulOp.result should matchPattern {
      case Value(IntegerType(IntData(25), Unsigned)) =>
    }
    adtMulOp.randProp should matchPattern { case Property(StringData("what")) =>
    }
  }

  "ADTOp test for correctly passing around the same instances" should "test all fields of a ADTOp" in {
    val opT = MLIRTrait.derived[MulFull]

    val op = TestCases.adtMulOpAllFields

    val unverMulOp = opT.unverify(op)
    val adtMulOp = opT.verify(unverMulOp)

    adtMulOp.operand1 equals unverMulOp.operands(0) should be(true)
    adtMulOp.operand2 equals unverMulOp.operands(1) should be(true)
    adtMulOp.result1 equals unverMulOp.results(0) should be(true)
    adtMulOp.result2 equals unverMulOp.results(1) should be(true)
    adtMulOp.randProp1.typ equals unverMulOp.dictionaryProperties(
      "randProp1"
    ) should be(true)
    adtMulOp.randProp2.typ equals unverMulOp.dictionaryProperties(
      "randProp2"
    ) should be(true)
    adtMulOp.reg1 equals unverMulOp.regions(0) should be(true)
    adtMulOp.reg2 equals unverMulOp.regions(1) should be(true)
    adtMulOp.succ1 equals unverMulOp.successors(0) should be(true)
    adtMulOp.succ2 equals unverMulOp.successors(1) should be(true)
  }

  "Single Variadic Conversion to ADTOp" should "Correctly translate from Single Variadic Unverified operation to ADT Operation" in {
    val opT = summon[MLIRTrait[MulSingleVariadic]]

    val op = TestCases.unverMulSinVarOp
    val adtMulSinVarOp = opT.verify(op)

    adtMulSinVarOp.lhs should matchPattern {
      case Value(IntegerType(IntData(5), Unsigned)) =>
    }
    adtMulSinVarOp.rhs should matchPattern {
      case List(
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned))
          ) =>
    }
    adtMulSinVarOp.result should matchPattern {
      case List(
            Result(IntegerType(IntData(25), Unsigned)),
            Result(IntegerType(IntData(25), Unsigned))
          ) =>
    }
    adtMulSinVarOp.randProp should matchPattern {
      case Property(StringData("what")) =>
    }
  }

  "Single Variadic Conversion to Unverified" should "Correctly translate from Single Variadic ADT operation to Univerified Operation" in {
    val opT = summon[MLIRTrait[MulSingleVariadic]]

    val op = TestCases.adtMulSinVarOp
    val unverMulSinVarOp = opT.unverify(op)

    unverMulSinVarOp.name should be("cmath.mulsinglevariadic")
    unverMulSinVarOp.operands should matchPattern {
      case ListType(
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned))
          ) =>
    }
    unverMulSinVarOp.results should matchPattern {
      case ListType(
            Result(IntegerType(IntData(25), Unsigned)),
            Result(IntegerType(IntData(25), Unsigned))
          ) =>
    }
    unverMulSinVarOp.dictionaryProperties("randProp") should matchPattern {
      case StringData("what") =>
    }
  }

  "Multi Variadic Conversion to ADTOp" should "Correctly translate from Multi Variadic Unverified operation to ADT Operation" in {
    val opT = summon[MLIRTrait[MulMultiVariadic]]

    val op = TestCases.unverMulMulVarOp
    val adtMulMulVarOp = opT.verify(op)

    adtMulMulVarOp.lhs should matchPattern {
      case Value(IntegerType(IntData(5), Unsigned)) =>
    }
    adtMulMulVarOp.rhs should matchPattern {
      case List(
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned))
          ) =>
    }
    adtMulMulVarOp.mhs should matchPattern {
      case List(
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned))
          ) =>
    }
    adtMulMulVarOp.result should matchPattern {
      case List(
            Result(IntegerType(IntData(25), Unsigned)),
            Result(IntegerType(IntData(25), Unsigned)),
            Result(IntegerType(IntData(25), Unsigned))
          ) =>
    }
    adtMulMulVarOp.result2 should matchPattern {
      case Result(IntegerType(IntData(25), Unsigned)) =>
    }
    adtMulMulVarOp.result3 should matchPattern {
      case List(
            Result(IntegerType(IntData(25), Unsigned)),
            Result(IntegerType(IntData(25), Unsigned))
          ) =>
    }
  }

  "Multi Variadic Conversion to Unverified" should "Correctly translate from Multi Variadic ADT operation to Univerified Operation" in {
    val opT = summon[MLIRTrait[MulMultiVariadic]]

    val op = TestCases.adtMulMulVarOp
    val unverMulSinVarOp = opT.unverify(op)

    unverMulSinVarOp.name should be("cmath.mulmultivariadic")
    unverMulSinVarOp.operands should matchPattern {
      case ListType(
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned)),
            Value(IntegerType(IntData(5), Unsigned))
          ) =>
    }
    unverMulSinVarOp.results should matchPattern {
      case ListType(
            Result(IntegerType(IntData(5), Unsigned)),
            Result(IntegerType(IntData(5), Unsigned)),
            Result(IntegerType(IntData(5), Unsigned)),
            Result(IntegerType(IntData(5), Unsigned)),
            Result(IntegerType(IntData(5), Unsigned)),
            Result(IntegerType(IntData(5), Unsigned))
          ) =>
    }
  }

}

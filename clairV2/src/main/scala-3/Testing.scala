package scair.clairV2.testing

import scair.clairV2.mirrored.*
import scair.dialects.builtin.*
import scair.ir.*
import scair.scairdl.constraints.BaseAttr
import scair.scairdl.constraints.ConstraintContext

// object ExampleWithGenedCode {

//   import test.generated.{x, y}

//   object Mul extends ADTCompanion {
//     val getMLIRRealm: MLIRRealm[Mul] = x
//   }

//   object Norm extends ADTCompanion {
//     val getMLIRRealm: MLIRRealm[Norm] = y
//   }

//   case class Mul(
//       lhs: Operand[IntegerAttr],
//       rhs: Operand[IntegerAttr],
//       result: Result[IntegerAttr]
//   ) extends ADTOperation

//   case class Norm(
//       norm: Operand[IntegerAttr],
//       result: Result[IntegerAttr]
//   ) extends ADTOperation

//   val ctxDict = Map(
//     "mul" -> Mul,
//     "norm" -> Norm
//   )

//   def main(args: Array[String]): Unit = {

//     // ================== //
//     //     Testing Mul    //
//     // ================== //

//     val regOpMul = RegisteredOperation(
//       name = "mul",
//       operands = ListType(
//         Value(IntegerAttr(IntData(1), IntegerType(IntData(32), Unsigned))),
//         Value(IntegerAttr(IntData(1), IntegerType(IntData(32), Unsigned)))
//       ),
//       results_types = ListType(
//         IntegerAttr(IntData(1), IntegerType(IntData(32), Unsigned))
//       )
//     )

//     val objMul = ctxDict("mul")

//     val mlirRealmMul = objMul.getMLIRRealm

//     val adtOpMul = mlirRealmMul.verify(regOpMul)
//     val regOpMulBack = mlirRealmMul.unverify(adtOpMul)

//     println("=============")
//     println("Testing Mul: ")
//     println(adtOpMul)
//     println(mlirRealmMul.verify(regOpMulBack))

//     // =================== //
//     //     Testing Norm    //
//     // =================== //

//     val regOpNorm = RegisteredOp[Norm](
//       name = "norm",
//       operands = ListType(
//         Value(IntegerAttr(IntData(1), IntegerType(IntData(32), Unsigned)))
//       ),
//       results_types = ListType(
//         IntegerAttr(IntData(1), IntegerType(IntData(32), Unsigned))
//       )
//     )

//     val objNorm = ctxDict("norm")

//     val mlirRealmNorm = objNorm.getMLIRRealm

//     val adtOpNorm = mlirRealmNorm.verify(regOpNorm)
//     val regOpNormBack = mlirRealmNorm.unverify(adtOpNorm)

//     println("=============")
//     println("Testing Norm: ")
//     println(adtOpNorm)
//     println(mlirRealmNorm.verify(regOpNormBack))
//     println("=============")
//   }

// }

object ExampleCodeGen {

  case class Mul(
      lhs: Operand[IntegerAttr],
      rhs: Operand[IntegerAttr],
      result: Result[IntegerAttr],
      randProp: Property[StringData]
  ) extends ADTOperation

  case class Norm(
      norm: Operand[IntegerAttr],
      result: Result[IntegerAttr]
  ) extends ADTOperation

  def main(args: Array[String]): Unit = {
    val mlirOpDef = summonMLIROps[(Mul, Norm)]

    println(mlirOpDef.print)
  }

}

object ConcreteExample {

  import scair.ir.ValueConversions.{resToVal, valToRes}

  val x = new MLIRRealm[Mul] {

    def unverify(op: Mul): RegisteredOperation = {
      val op1 = RegisteredOperation(
        name = "mul",
        operands = ListType(
          op.lhs,
          op.rhs
        )
      )

      op1.results.clear()
      op1.results.addAll(ListType(op.result))
      op1
    }

    def verify(op: RegisteredOperation): Mul = {

      if (op.operands.length != 2) then
        throw new Exception(s"Expected 2 operands, got ${op.operands.length}")

      if (op.results.length != 1) then
        throw new Exception(s"Expected 1 operands, got ${op.operands.length}")

      BaseAttr[IntegerAttr]()
        .verify(op.operands(0).typ, new ConstraintContext())

      BaseAttr[IntegerAttr]()
        .verify(op.operands(1).typ, new ConstraintContext())

      BaseAttr[IntegerAttr]()
        .verify(op.results(0).typ, new ConstraintContext())

      Mul(
        op.operands(0).asInstanceOf[Value[IntegerAttr]],
        op.operands(1).asInstanceOf[Value[IntegerAttr]],
        op.results(0).asInstanceOf[Value[IntegerAttr]]
      )
    }

  }

  object Mul {
    given MLIRRealm[Mul] = x
    val getMLIRRealm: MLIRRealm[Mul] = summon[MLIRRealm[Mul]]
  }

  case class Mul(
      lhs: Operand[IntegerAttr],
      rhs: Operand[IntegerAttr],
      result: Result[IntegerAttr]
  ) extends ADTOperation

  val ctxDict = Map("mul" -> Mul)

  def main(args: Array[String]): Unit = {

    val regOp = RegisteredOperation(
      name = "mul",
      operands = ListType(
        Value(IntegerAttr(IntData(1), IntegerType(IntData(32), Unsigned))),
        Value(IntegerAttr(IntData(1), IntegerType(IntData(32), Unsigned)))
      ),
      results_types = ListType(
        IntegerAttr(IntData(1), IntegerType(IntData(32), Unsigned))
      )
    )

    val obj = ctxDict("mul")

    val a = obj.getMLIRRealm

    val adtOp = a.verify(regOp)
    val regOp2 = a.unverify(adtOp)

    println(adtOp)
    println(a.verify(regOp2))
  }

}

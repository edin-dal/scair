// RUN: scala full-classpath %s - | filecheck %s

import scair.dialects.cmathv2.NormV2Helper
import scair.dialects.builtin.{IntegerAttr, IntData, IntegerType, Unsigned}
import scair.ir.*
import scair.Printer

object Main {

  val x = ListType[Value[Attribute]](
    Value(IntegerAttr(IntData(5), IntegerType(IntData(3), Unsigned)))
  )

  val y = ListType[Attribute](
    IntegerAttr(IntData(5), IntegerType(IntData(3), Unsigned))
  )

  def main(args: Array[String]): Unit = {

    val printer = new Printer()

    val mlirRealm = NormV2Helper.getMLIRRealm

    val unVerOP = mlirRealm.constructUnverifiedOp(
      operands = x,
      results_types = y
    )

    val adtOP = mlirRealm.verify(unVerOP)

    val backToUnVerOP = mlirRealm.unverify(adtOP)
    println(backToUnVerOP.print(printer))

  }

}

// CHECK: %0 = "normv2"(%1) : (5 : ui3) -> (5 : ui3)

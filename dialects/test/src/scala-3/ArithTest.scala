package scair

import scair.ir.*
import scair.dialects.arith.*
import scair.dialects.builtin.*
import scair.dialects.func.*
import scair.utils.*
import scair.dialects.testutils.IRTestKit.*

import org.scalatest.*
import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import scair.transformations.RewriteMethods

/** Regression test for generic rewrite utilities (Replace / Erase) on a small
  * arith+func program.
  */
final class ArithRewriteInfraSpec extends AnyFlatSpec with BeforeAndAfter:

  given indentLevel: Int = 0

  "RewriteMethods" should "replace and erase ops while keeping IR consistent" in {
    val zero = scair.dialects.arith.Constant(
      IntegerAttr(IntData(0), I32),
      Result(I32),
    )
    val module = ModuleOp(
      Region(
        Seq(
          Block(
            operations = Seq(
              Func(
                "suchCompute",
                FunctionType(Seq(I32), Seq(I32)),
                None,
                Region(
                  Seq(
                    Block(
                      I32,
                      (arg: Value[?]) =>
                        val addres = Result(I32)
                        val lhs = arg.asInstanceOf[Value[IntegerType]]
                        val rhs =
                          zero.results.head.asInstanceOf[Result[IntegerType]]
                        val add = AddI(lhs, rhs, addres)
                        val ret = Return(Seq(addres))
                        Seq(zero, add, ret),
                    )
                  )
                ),
              )
            )
          )
        )
      )
    )
    printIR(module) shouldEqual """
builtin.module {
  func.func @suchCompute(%0: i32) -> i32 {
    %1 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %2 = "arith.addi"(%0, %1) : (i32, i32) -> i32
    func.return %2 : i32
  }
}
""".trim()

    val func = module.regions.head.blocks.head.operations.head
    val arg = func.regions.head.blocks.head.arguments.head
    val BlockOperations(_, add, ret) = func.regions.head.blocks.head.operations
    RewriteMethods.replaceOp(add, Seq(), Some(Seq(arg)))
    RewriteMethods.eraseOp(zero)

    printIR(module.structured.get) shouldEqual """
builtin.module {
  func.func @suchCompute(%0: i32) -> i32 {
    func.return %0 : i32
  }
}
""".trim()
  }

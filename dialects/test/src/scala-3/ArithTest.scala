package scair

import scair.ir.*
import scair.dialects.arith.*
import scair.dialects.builtin.*
import scair.dialects.func.*
import scair.Printer

import org.scalatest.*
import org.scalatest.flatspec.*
import org.scalatest.matchers.should.Matchers.*
import scair.transformations.RewriteMethods
import java.io.*

/** In case not clear from style - not a serious test, just temporarily testing
  * new infra at time of writing. Please remove if this seems useless at the
  * time of reading ;)
  */

class ArithTests extends AnyFlatSpec with BeforeAndAfter {

  given indentLevel: Int = 0

  "Such real ADT manipulation" should "flex how working it is" in {
    val zero = UnregisteredOperation(
      name = "arith.constant",
      results = Seq(Result(I32)),
      properties = Map("value" -> IntegerAttr(IntData(0), I32))
    )
    //   val and =
    val module = ModuleOp(regions =
      Seq(Region(Seq(Block(operations =
        Seq(Func(
          "suchCompute",
          FunctionType(Seq(I32), Seq(I32)),
          None,
          Region(Seq(Block(
            I32,
            (arg: Value[?]) =>
              val addres = Result(I32)
              val lhs = arg.asInstanceOf[Value[IntegerType]]
              val rhs = zero.results.head.asInstanceOf[Result[IntegerType]]
              val add = AddI(lhs, rhs, addres)
              val ret = Return(Seq(addres))
              Seq(zero, add, ret)
          )))
        ))
      ))))
    )
    var out = StringWriter()
    Printer(p = PrintWriter(out)).print(module)
    out.toString().trim() shouldEqual """
builtin.module {
  "func.func"() <{sym_name = "suchCompute", function_type = (i32) -> i32}> ({
  ^bb0(%0: i32):
    %1 = "arith.constant"() <{value = 0 : i32}> : () -> (i32)
    %2 = "arith.addi"(%0, %1) : (i32, i32) -> (i32)
    "func.return"(%2) : (i32) -> ()
  }) : () -> ()
}
""".trim()

    val func = module.regions.head.blocks.head.operations.head
    val arg = func.regions.head.blocks.head.arguments.head
    val ListType(_, add, ret) = func.regions.head.blocks.head.operations
    RewriteMethods.replace_op(add, Seq(), Some(Seq(arg)))
    RewriteMethods.erase_op(zero)

    out = StringWriter()
    Printer(p = PrintWriter(out)).print(module)
    out.toString().trim() shouldEqual """
builtin.module {
  "func.func"() <{sym_name = "suchCompute", function_type = (i32) -> i32}> ({
  ^bb0(%0: i32):
    "func.return"(%0) : (i32) -> ()
  }) : () -> ()
}
""".trim()
  }

}

package scair.interpreter

import scair.dialects.memref
import scair.interpreter.ShapedArray

object run_alloc extends OpImpl[memref.Alloc]:

  def compute(
      alloc_op: memref.Alloc,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Tuple,
  ): Any =

    // initialising a zero array to represent allocated memory
    // multi-dimensional objects are packed into a 1-D array
    args match
      case EmptyTuple      => ShapedArray(Array(0), Seq(1)) // 0-D memref
      case Tuple1(indices) =>
        ShapedArray(
          Array.fill(indices.asInstanceOf[Seq[Int]].product)(0),
          indices.asInstanceOf[Seq[Int]],
        )
      case _ => throw new Exception("Alloc operands must be Seq[Int]")

object run_store extends OpImpl[memref.Store]:

  def compute(
      store_op: memref.Store,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Tuple,
  ): Any =
    args match
      case (value: Int, memref: ShapedArray) =>
        memref(Seq(0)) = value // storing in first index for 0-D memref
      case (value: Int, memref: ShapedArray, indices) =>
        memref(indices.asInstanceOf[Seq[Int]]) =
          value // storing in specified index for higher-D memref
      case _ =>
        throw new Exception(
          "Store operands must be (Int, ShapedArray) or (Int, ShapedArray, Seq[Int])"
        )

object run_load extends OpImpl[memref.Load]:

  def compute(
      load_op: memref.Load,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Tuple,
  ): Any =
    args match
      case Tuple1(memref: ShapedArray) =>
        memref(Seq(0))
      case (memref: ShapedArray, indices) =>
        memref(indices.asInstanceOf[Seq[Int]])
      case _ =>
        throw new Exception("Load operands must be (ShapedArray, Seq[Int])")

// constructing memref dialect
val InterpreterMemrefDialect: InterpreterDialect =
  Seq(
    run_alloc,
    run_store,
    run_load,
  )

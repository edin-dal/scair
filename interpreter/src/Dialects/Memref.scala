package scair.interpreter

import scair.dialects.memref
import scair.interpreter.ShapedArray

// TODO: symbolOperand case
object run_alloc extends OpImpl[memref.Alloc]:

  def compute(
      alloc_op: memref.Alloc,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Seq[Any],
  ): Option[Any] =

    // initialising a zero array to represent allocated memory
    // multi-dimensional objects are packed into a 1-D array
    args match
      case Seq()          => Some(ShapedArray(Seq(1))) // 0-D memref
      case Seq(size: Int) => // 1-D memref
        Some(
          ShapedArray(
            Seq(size)
          )
        )
      case Seq(sizes*) => // multi-D memref
        Some(
          ShapedArray(
            sizes.asInstanceOf[Seq[Int]] // TODO: some way to remove asInstance
          )
        )

object run_store extends OpImpl[memref.Store]:

  def compute(
      store_op: memref.Store,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Seq[Any],
  ): Option[Any] =
    args match
      case Seq(value: Int, memref: ShapedArray) =>
        memref(Seq(0)) = value // storing in first index for 0-D memref
        None
      case Seq(value: Int, memref: ShapedArray, indices*) =>
        memref(indices.asInstanceOf[Seq[Int]]) =
          value // storing in specified index for higher-D memref
        None
      case _ =>
        throw new Exception(
          "Store operands must be (Int, ShapedArray) or (Int, ShapedArray, Seq[Int])"
        )

object run_load extends OpImpl[memref.Load]:

  def compute(
      load_op: memref.Load,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
      args: Seq[Any],
  ): Option[Any] =
    args match
      case Seq(memref: ShapedArray) =>
        Some(memref(Seq(0)))
      case Seq(memref: ShapedArray, indices*) =>
        Some(memref(indices.asInstanceOf[Seq[Int]]))
      case _ =>
        throw new Exception("Load operands must be (ShapedArray, Seq[Int])")

// constructing memref dialect
val InterpreterMemrefDialect: InterpreterDialect =
  Seq(
    run_alloc,
    run_store,
    run_load,
  )

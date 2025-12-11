package scair.interpreter

import scair.dialects.memref
import scair.interpreter.ShapedArray

object run_alloc extends OpImpl[memref.Alloc]:

  def run(
      alloc_op: memref.Alloc,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
  ): Unit =

    // retrieving the Seq[int] that derives the dimension of the the array and thus memory
    val shapeSeq =
      for dim <- alloc_op.dynamicSizes yield interpreter.lookup_op(dim, ctx)

    // initialising a zero array to represent allocated memory
    // multi-dimensional objects are packed into a 1-D array
    ctx.vars.put(
      alloc_op.memref,
      ShapedArray(Array.fill(shapeSeq.product)(0), shapeSeq),
    )

object run_store extends OpImpl[memref.Store]:

  def run(
      store_op: memref.Store,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
  ): Unit =
    val value = interpreter.lookup_op(store_op.value, ctx)
    val memref = interpreter.lookup_op(store_op.memref, ctx)
    // could already be a list?
    val indices =
      for index <- store_op.indices yield interpreter.lookup_op(index, ctx)
    memref(indices) = value

object run_load extends OpImpl[memref.Load]:

  def run(
      load_op: memref.Load,
      interpreter: Interpreter,
      ctx: RuntimeCtx,
  ): Unit =
    val memref = interpreter.lookup_op(load_op.memref, ctx)
    val indices =
      for index <- load_op.indices yield interpreter.lookup_op(index, ctx)
    ctx.vars.put(load_op.result, memref(indices))

// constructing memref dialect
val InterpreterMemrefDialect: InterpreterDialect =
  Seq(
    run_alloc,
    run_store,
    run_load,
  )

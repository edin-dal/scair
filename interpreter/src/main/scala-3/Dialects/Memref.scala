package scair.tools

import scair.dialects.memref

import scala.reflect.ClassTag

def run_alloc(alloc_op: memref.Alloc, ctx: RuntimeCtx): Unit =

  // retrieving the Seq[int] that derives the dimension of the the array and thus memory
  val shape_seq = for dim <- alloc_op.dynamicSizes yield lookup_op(dim, ctx)

  // initialising a zero array to represent allocated memory
  // multi-dimensional objects are packed into a 1-D array
  ctx.vars.put(alloc_op.memref, ShapedArray(Array.fill(shape_seq.product)(0), shape_seq))

def run_store(store_op: memref.Store, ctx: RuntimeCtx): Unit =
  val value = lookup_op(store_op.value, ctx).asInstanceOf[Int]
  val memref = lookup_op(store_op.memref, ctx)
  // could already be a list?
  val indices = for index <- store_op.indices yield lookup_op(index, ctx)
  memref(indices) = value

def run_load(load_op: memref.Load, ctx: RuntimeCtx): Unit =
  val memref = lookup_op(load_op.memref, ctx).asInstanceOf[ShapedArray]
  val indices = for index <- load_op.indices yield lookup_op(index, ctx)
  ctx.vars.put(load_op.result, memref(indices))

// constructing memref dialect
val InterpreterMemrefDialect = summonImplementations(
  Seq(
    OpImpl(run_alloc),
    OpImpl(run_store),
    OpImpl(run_load)
  )
)

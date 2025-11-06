package scair.tools

import scair.dialects.memref

import scala.reflect.ClassTag

trait MemoryHandler:
  self: Interpreter =>

  def allocate_memory(alloc_op: memref.Alloc, ctx: InterpreterCtx): Unit =

    // retrieving the Seq[int] that derives the dimension of the the array and thus memory
    val shape_seq: Seq[Int] = alloc_op.dynamicSizes.map { dim =>
      lookup_op(dim, ctx) match
        // ensuring all elements are int
        case i: Int => i
        case other  =>
          throw new Exception(
            "Expected int arguments for array shape"
          )
    }

    // initialising a zero array to represent allocated memory
    // multi-dimensional objects are packed into a 1-D array
    val shaped_array =
      ShapedArray[Int](Array.fill(shape_seq.product)(0), shape_seq)
    ctx.vars.put(alloc_op.memref, shaped_array)

  def store_memory(store_op: memref.Store, ctx: InterpreterCtx): Unit =
    val value = lookup_op(store_op.value, ctx)
    val memref = lookup_op(store_op.memref, ctx)
    // could already be a list?
    val indices = for index <- store_op.indices yield lookup_op(index, ctx)
    val int_indices = indices.collect { case i: Int => i }

    (memref, value) match
      case (sa: ShapedArray[?], value: Int)
          if sa.tag == implicitly[ClassTag[Int]] =>
        val int_sa = sa.asInstanceOf[ShapedArray[Int]]
        int_sa(int_indices) = value
      case _ =>
        throw new Exception(
          "Memory reference points to invalid memory data type"
        )

  def load_memory(load_op: memref.Load, ctx: InterpreterCtx): Unit =
    val memref = lookup_op(load_op.memref, ctx)
    val indices = for index <- load_op.indices yield lookup_op(index, ctx)
    val int_indices = indices.collect { case i: Int => i }

    memref match
      case sa: ShapedArray[?] if sa.tag == implicitly[ClassTag[Int]] =>
        val int_sa = sa.asInstanceOf[ShapedArray[Int]]
        ctx.vars.put(load_op.result, int_sa(int_indices))
      case _ =>
        throw new Exception(
          "Memory reference points to invalid memory data type"
        )

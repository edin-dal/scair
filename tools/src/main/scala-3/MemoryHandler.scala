package scair.tools

import scair.dialects.memref
import scala.collection.mutable.ListBuffer

trait MemoryHandler:
  self: Interpreter =>

  // assign some attribute to variables
  def allocate_memory(alloc_op: memref.Alloc, ctx: InterpreterCtx): Unit =
    // TODO: consider 3d array and indices
    val shape_seq: Seq[Int] = alloc_op.dynamicSizes.map { dim =>
      lookup_op(dim, ctx) match {
        case i: Int => i
        case other  => throw new Exception(
          "Expected int arguments for array shape"
        )
      }
    }


    val shaped_array = ShapedArray[Int](Array.range(0, shape_seq.product), shape_seq)
    ctx.vars.put(alloc_op.memref, shaped_array)

  def store_memory(store_op: memref.Store, ctx: InterpreterCtx): Unit = 
    val value_to_store = lookup_op(store_op.value, ctx)
    val memref = lookup_op(store_op.memref, ctx)
    // addr match
    //   case integerAddr: IntegerAttr =>
    //     val memAddrIndex = integerAddr.value.value.toInt
    //     if ctx.memory.nonEmpty && ctx.memory.isDefinedAt(0) then
    //       ctx.memory(memAddrIndex) = valueToStore
    //     else ctx.memory += valueToStore
    //   case _ => throw new Exception("non integer addresses not yet supported")

  // TODO: index accesses
  def load_memory(load_op: memref.Load, ctx: InterpreterCtx): Unit = 0
    // val memoryOperand = load_op.memref.owner.getOrElse(
    //   throw new Exception("memref operand for store operation not found")
    // )
    // val addr = lookup_op(memoryOperand, ctx)
// 
    // addr match
    //   case integerAddr: IntegerAttr =>
    //     val memAddrIndex = integerAddr.value.value.toInt
    //     // note that load and store values will not be connected even with a shared value
    //     val retrievedVal = ctx.memory(memAddrIndex)
    //     ctx.vars.put(load_op, retrievedVal)
    //   case _ => throw new Exception("non integer addresses not yet supported")

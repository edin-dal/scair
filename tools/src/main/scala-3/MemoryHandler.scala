package scair.tools

import scair.dialects.memref
import scair.dialects.builtin.IntegerAttr
import scair.dialects.builtin.I32
import scair.dialects.builtin.IntData

trait MemoryHandler { self: Interpreter =>

    // does not consider freed memory space rn
    private var nextAvailableAddr: Int = 0

    // assign some attribute to variables
    def allocate_memory(alloc_op: memref.Alloc, ctx: InterpreterCtx): Unit = {
        // TODO: expand for other types
        // alloc_op.memref.typ.elementType match {
        //      case intType: IntegerType =>
        //          width = intType.width.value.toInt
        //      case _ => throw new Exception("Unsupported attribute type for alloc operation")
        // }
        val addrAttr = IntegerAttr(IntData(nextAvailableAddr), I32)
        ctx.vars.put(alloc_op, addrAttr)
        nextAvailableAddr += 1
    }

    def store_memory(store_op: memref.Store, ctx: InterpreterCtx): Unit = {
        val valueOperand = store_op.value.owner.getOrElse(throw new Exception("value operand for store operation not found"))
        val valueToStore = lookup_op(valueOperand, ctx)
        val memoryOperand = store_op.memref.owner.getOrElse(throw new Exception("memref operand for store operation not found"))

        // lookup operand of memory to find value needed for address
        val addr = lookup_op(memoryOperand, ctx)
        addr match {
            case integerAddr: IntegerAttr =>
                val memAddrIndex = integerAddr.value.value.toInt
                if (ctx.memory.nonEmpty && ctx.memory.isDefinedAt(0)) {
                    ctx.memory(memAddrIndex) = valueToStore
                } else {
                    ctx.memory += valueToStore
                }
            case _ => throw new Exception("non integer addresses not yet supported")
        }
    }

    // TODO: index accesses
    def load_memory(load_op: memref.Load, ctx: InterpreterCtx): Unit = {
        val memoryOperand = load_op.memref.owner.getOrElse(throw new Exception("memref operand for store operation not found"))
        val addr = lookup_op(memoryOperand, ctx)

        addr match {
            case integerAddr: IntegerAttr =>
                val memAddrIndex = integerAddr.value.value.toInt
                // note that load and store values will not be connected even with a shared value
                val retrievedVal = ctx.memory(memAddrIndex)
                ctx.vars.put(load_op, retrievedVal)
            case _ => throw new Exception("non integer addresses not yet supported")
        }
    }
}
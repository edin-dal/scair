package scair.tools

import scair.ir.*

// TODO: move all interpreter helper functions here
// match owner of operation to its evaluated value and return it

def lookup_op(operation: Operation | Block, ctx: InterpreterCtx): Attribute = {
    operation match {
        case op: Operation =>
            ctx.vars.getOrElse(op, throw new Exception(s"Operation ${op} not found in context"))
        case block: Block =>
            throw new Exception("Block lookup not supported yet")
        case _ => throw new Exception(s"Cannot find evaluation for ${operation}")
    }
}

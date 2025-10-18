package scair.tools

import scair.ir.*

// TODO: move all interpreter helper functions here
// match owner of operation to its evaluated value and return it

def find_evaluation(operation: Operation | Block, ctx: InterpreterCtx): Option[Attribute] = {
    operation match {
        case op: Operation =>
            ctx.vars.get(op)
        case _ => None
    }
}

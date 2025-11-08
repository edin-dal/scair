package scair.tools

import scair.ir.*

// lookup function for context variables
def lookup_op(value: Value[Attribute], ctx: RuntimeCtx): Any =
  ctx.vars.getOrElse(
    value,
    throw new Exception(s"Value $value} not found in context")
  )

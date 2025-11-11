package scair.tools

import scair.ir.*
import scair.dialects.builtin.*
import scala.collection.mutable

val interpreterContext = InterpreterContext(
      Seq(
        InterpreterFuncDialect,
        InterpreterArithDialect,
        InterpreterMemrefDialect
      ),
      mutable.Map()
    )

// type maps from operation to its resulting lookup type
// base case is Int (may have issues later)
type ImplOf[T <: Value[Attribute]] = T match
  case Value[MemrefType] => ShapedArray
  case _ => Int

// lookup function for context variables
// does not work for Bool-like vals due to inability to prove disjoint for ImplOf
def lookup_op[T <: Value[Attribute]](value: T, ctx: RuntimeCtx): ImplOf[T] =
  ctx.vars.get(value) match
    case Some(v) => v.asInstanceOf[ImplOf[T]]
    case _ => throw new Exception(s"Variable ${value} not found in context")
  
def lookup_boollike(value: Value[Attribute], ctx: RuntimeCtx): Int =
  ctx.vars.get(value) match
    case Some(v: Int) => v
    case _ => throw new Exception(s"Bool-like ${value} not found in context")

// helper function to summon implementations into an InterpreterDialect
inline def summonImplementations(
    impls: Seq[OpImpl[? <: Operation]]
): InterpreterDialect =
  new InterpreterDialect(impls)
package scair.interpreter

import scair.dialects.builtin.*
import scair.dialects.func
import scair.interpreter.ShapedArray
import scair.ir.*

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.reflect.ClassTag

// global implementation dictionary for interpreter
val impl_dict = mutable
  .Map[Class[? <: Operation], OpImpl[
    ? <: Operation
  ]]()

// global external function dictionary for interpreter
val ext_func_dict = mutable.Map[String, FunctionCtx]()

// OpImpl class representing operation implementation, mainly for accessing implementation type information
trait OpImpl[T <: Operation: ClassTag]:
  def opType: Class[T] = summon[ClassTag[T]].runtimeClass.asInstanceOf[Class[T]]
  def run(op: T, interpreter: Interpreter, ctx: RuntimeCtx): Unit

// function context class for saving a context at the function's definition (cannot use values defined after the function itself)
case class FunctionCtx(
    name: String,
    saved_ctx: RuntimeCtx,
    body: Block,
)

// interpreter context class stores variables, function definitions and the current result
class RuntimeCtx(
    val vars: mutable.Map[Value[Attribute], Any],
    val funcs: ListBuffer[FunctionCtx],
    var result: Option[Any] = None,
):

  // creates independent context
  def deep_clone_ctx(): RuntimeCtx =
    RuntimeCtx(
      mutable.Map() ++ this.vars,
      ListBuffer() ++ this.funcs,
      None,
    )

  // helper function for adding a function to the context
  def add_func_ctx(function: func.Func): Unit =
    val func_ctx = FunctionCtx(
      name = function.sym_name,
      saved_ctx = this.deep_clone_ctx(),
      body = function.body.blocks.head,
    )
    this.funcs.append(func_ctx)

//
// ██╗ ███╗░░██╗ ████████╗ ███████╗ ██████╗░ ██████╗░ ██████╗░ ███████╗ ████████╗ ███████╗ ██████╗░
// ██║ ████╗░██║ ╚══██╔══╝ ██╔════╝ ██╔══██╗ ██╔══██╗ ██╔══██╗ ██╔════╝ ╚══██╔══╝ ██╔════╝ ██╔══██╗
// ██║ ██╔██╗██║ ░░░██║░░░ █████╗░░ ██████╔╝ ██████╔╝ ██████╔╝ █████╗░░ ░░░██║░░░ █████╗░░ ██████╔╝
// ██║ ██║╚████║ ░░░██║░░░ ██╔══╝░░ ██╔══██╗ ██╔═══╝░ ██╔══██╗ ██╔══╝░░ ░░░██║░░░ ██╔══╝░░ ██╔══██╗
// ██║ ██║░╚███║ ░░░██║░░░ ███████╗ ██║░░██║ ██║░░░░░ ██║░░██║ ███████╗ ░░░██║░░░ ███████╗ ██║░░██║
// ╚═╝ ╚═╝░░╚══╝ ░░░╚═╝░░░ ╚══════╝ ╚═╝░░╚═╝ ╚═╝░░░░░ ╚═╝░░╚═╝ ╚══════╝ ░░░╚═╝░░░ ╚══════╝ ╚═╝░░╚═╝
//

// INTERPRETER CLASS
class Interpreter:

  // type maps from operation to its resulting lookup type
  // base case is Int (may have issues later)
  type ImplOf[T <: Value[Attribute]] = T match
    case Value[MemrefType] => ShapedArray
    case _                 => Int

  // lookup function for context variables
  // does not work for Bool-like vals due to inability to prove disjoint for ImplOf
  def lookup_op[T <: Value[Attribute]](value: T, ctx: RuntimeCtx): ImplOf[T] =
    ctx.vars.get(value) match
      case Some(v) => v.asInstanceOf[ImplOf[T]]
      case _ => throw new Exception(s"Variable $value not found in context")

  def lookup_boollike(value: Value[Attribute], ctx: RuntimeCtx): Int =
    ctx.vars.get(value) match
      case Some(v: Int) => v
      case _ => throw new Exception(s"Bool-like $value not found in context")

  def register_implementations(): Unit =
    for dialect <- allInterpreterDialects do
      for impl <- dialect do impl_dict.put(impl.opType, impl)

  // keeping buffer function for extensibility
  def interpret(block: Block, ctx: RuntimeCtx): Option[Any] =
    for op <- block.operations do interpret_op(op, ctx)
    ctx.result

  // note: results are put within implementations, may change later
  def interpret_op(op: Operation, ctx: RuntimeCtx): Unit =
    val impl = impl_dict.get(op.getClass)
    impl match
      case Some(impl) => impl.asInstanceOf[OpImpl[Operation]].run(op, this, ctx)
      case None       =>
        throw new Exception("Unsupported operation when interpreting")

package scair.interpreter

import scair.dialects.builtin.*
import scair.interpreter.ShapedArray
import scair.ir.*

import scala.collection.mutable
import scala.reflect.ClassTag

// global implementation dictionary for interpreter
val impl_dict = mutable
  .Map[Class[? <: Operation], OpImpl[
    ? <: Operation
  ]]()

// OpImpl class representing operation implementation, mainly for accessing implementation type information
trait OpImpl[T <: Operation: ClassTag]:
  def opType: Class[T] = summon[ClassTag[T]].runtimeClass.asInstanceOf[Class[T]]
  def run(op: T, interpreter: Interpreter, ctx: RuntimeCtx): Unit

// interpreter context class stores variables, function definitions and the current result
class RuntimeCtx(
    val scopedDict: ScopedDict,
    val symbols: mutable.Map[String, Operation], // for now operations only
    var result: Option[Any] = None,
):

  // creates independent context
  def push_scope(): RuntimeCtx =
    RuntimeCtx(
      ScopedDict(Some(this.scopedDict), mutable.Map()),
      mutable.Map() ++ this.symbols,
      None,
    )

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
    ctx.scopedDict.get(value) match
      case Some(v) => v.asInstanceOf[ImplOf[T]]
      case _ => throw new Exception(s"Variable $value not found in context")

  def lookup_boollike(value: Value[Attribute], ctx: RuntimeCtx): Int =
    ctx.scopedDict.get(value) match
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

  def interpreter_print(value: Any): Unit =
    value match
      case 0 => println("false")
      case 1 => println("true")
      case _ => println(value)

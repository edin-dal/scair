package scair.interpreter

import scair.dialects.builtin.*
import scair.dialects.func
import scair.interpreter.ShapedArray
import scair.ir.*

import scala.collection.mutable
import scala.reflect.ClassTag
import scair.dialects.arith.AnyIntegerType

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
    var result: Option[Any] = None,
):

  // creates new runtime ctx with new scope but shared symbol table
  def push_scope(): RuntimeCtx =
    RuntimeCtx(
      ScopedDict(Some(this.scopedDict), mutable.Map()),
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

class Interpreter(
    val module: ModuleOp,
    val symbolTable: mutable.Map[String, Operation] = mutable
      .Map(), // for now operations only
    val dialects: Seq[InterpreterDialect],
):

  initialize_interpreter()

  def initialize_interpreter(): Unit =
    register_implementations()
    get_symbols_from_module()

  def get_symbols_from_module(): Unit =
    for op <- module.body.blocks.head.operations do
      op match
        case func_op: func.Func =>
          // add function to symbol table if not main
          symbolTable.put(func_op.sym_name.stringLiteral, func_op)
        case _ => () // ignore other ops, global vars not yet supported prob...

  // type maps from operation to its resulting lookup type for type-safe lookups
  type TypeMap[T <: Value[Attribute]] = T match
    case Value[MemrefType]       => ShapedArray
    case Value[IntegerAttr]      => Int
    case Value[AnyIntegerType]   => Int
    case Value[FloatAttr]        => Double
    case Value[FloatData]        => Double
    case Value[FloatType]        => Double
    case _                       => Unit

  // lookup function for context variables
  // does not work for Bool-like vals due to inability to prove disjoint for TypeMap
  def lookup_op[T <: Value[Attribute]](value: T, ctx: RuntimeCtx)(using tm: TypeMapper[T]): TestTypeMap[T] =
    ctx.scopedDict.get(value) match
      case Some(v) => v.asInstanceOf[TestTypeMap[T]]
      case _ => throw new Exception(s"Variable $value not found in context")

  def lookup_boollike(value: Value[Attribute], ctx: RuntimeCtx): Int =
    ctx.scopedDict.get(value) match
      case Some(v: Int) => v
      case _ => throw new Exception(s"Bool-like $value not found in context")

  def register_implementations(): Unit =
    for dialect <- dialects do
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

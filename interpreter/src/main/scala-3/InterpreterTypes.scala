package scair.interpreter

import scair.dialects.func
import scair.ir.*

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.reflect.ClassTag

//
// ██╗ ███╗░░██╗ ████████╗ ███████╗ ██████╗░ ██████╗░ ██████╗░ ███████╗ ████████╗ ███████╗ ██████╗░
// ██║ ████╗░██║ ╚══██╔══╝ ██╔════╝ ██╔══██╗ ██╔══██╗ ██╔══██╗ ██╔════╝ ╚══██╔══╝ ██╔════╝ ██╔══██╗
// ██║ ██╔██╗██║ ░░░██║░░░ █████╗░░ ██████╔╝ ██████╔╝ ██████╔╝ █████╗░░ ░░░██║░░░ █████╗░░ ██████╔╝
// ██║ ██║╚████║ ░░░██║░░░ ██╔══╝░░ ██╔══██╗ ██╔═══╝░ ██╔══██╗ ██╔══╝░░ ░░░██║░░░ ██╔══╝░░ ██╔══██╗
// ██║ ██║░╚███║ ░░░██║░░░ ███████╗ ██║░░██║ ██║░░░░░ ██║░░██║ ███████╗ ░░░██║░░░ ███████╗ ██║░░██║
// ╚═╝ ╚═╝░░╚══╝ ░░░╚═╝░░░ ╚══════╝ ╚═╝░░╚═╝ ╚═╝░░░░░ ╚═╝░░╚═╝ ╚══════╝ ░░░╚═╝░░░ ╚══════╝ ╚═╝░░╚═╝
//
// ████████╗ ██╗░░░██╗ ██████╗░ ███████╗ ░██████╗
// ╚══██╔══╝ ╚██╗░██╔╝ ██╔══██╗ ██╔════╝ ██╔════╝
// ░░░██║░░░ ░╚████╔╝░ ██████╔╝ █████╗░░ ╚█████╗░
// ░░░██║░░░ ░░╚██╔╝░░ ██╔═══╝░ ██╔══╝░░ ░╚═══██╗
// ░░░██║░░░ ░░░██║░░░ ██║░░░░░ ███████╗ ██████╔╝
// ░░░╚═╝░░░ ░░░╚═╝░░░ ╚═╝░░░░░ ╚══════╝ ╚═════╝░
//

// interpreter context class stores variables, memory, function definitions and the current result
class InterpreterCtx(
    val vars: mutable.Map[Value[Attribute], Any],
    val funcs: ListBuffer[FunctionCtx],
    var result: Option[Any] = None
):

  // creates independent context
  def clone_ctx(): InterpreterCtx =
    InterpreterCtx(
      mutable.Map() ++ this.vars,
      ListBuffer() ++ this.funcs,
      None
    )

  // helper function for adding a function to the context
  def add_func_ctx(function: func.Func): Unit =
    val func_ctx = FunctionCtx(
      name = function.sym_name,
      saved_ctx = this.clone_ctx(),
      body = function.body.blocks.head
    )
    this.funcs.append(func_ctx)

// function context class for saving a context at the function's definition (cannot use values defined after the function itself)
case class FunctionCtx(
    name: String,
    saved_ctx: InterpreterCtx,
    body: Block
)

// custom ShapedArray class used for memory and other multi-dimensional data structures
case class ShapedArray[T: ClassTag](
    private val data: Array[T],
    shape: Seq[Int]
):

  lazy val strides: Seq[Int] =
    shape.scanRight(1)(_ * _).tail

  val tag: ClassTag[T] = implicitly[ClassTag[T]]

  def length: Int =
    shape.product

  private def offset(indices: Seq[Int]): Int =
    indices.zip(strides).map(_ * _).sum

  def apply(indices: Seq[Int]): T = data(offset(indices))

  def update(indices: Seq[Int], value: T): Unit =
    data(offset(indices)) = value

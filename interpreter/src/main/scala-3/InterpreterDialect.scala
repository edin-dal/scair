package scair.tools

import scair.ir.*

import scala.reflect.ClassTag
import scala.collection.mutable

// definition of all dialects in interpreter context
val interpreterContext = InterpreterContext(
      Seq(
        InterpreterFuncDialect,
        InterpreterArithDialect,
        InterpreterMemrefDialect
      ),
      mutable.Map()
    )

// OpImpl class representing operation implementation, mainly for accessing implementation type information
trait OpImpl[T <: Operation: ClassTag]:
    def opType: Class[T] = summon[ClassTag[T]].runtimeClass.asInstanceOf[Class[T]]
    def run(op: T, interpreter: Interpreter, ctx: RuntimeCtx): Unit

// sequence of OpIml to represent a dialect for interpretation
final case class InterpreterDialect(
    val implementations: Seq[OpImpl[? <: Operation]]
)



package scair.tools

import scair.ir.*

import scala.reflect.ClassTag

// OpImpl class representing operation implementation, mainly for accessing implementation type information
trait OpImpl[T <: Operation: ClassTag]:
    def opType: Class[T] = summon[ClassTag[T]].runtimeClass.asInstanceOf[Class[T]]
    def run(op: T, interpreter: Interpreter, ctx: RuntimeCtx): Unit

package scair.tools

import scair.ir.*

import scala.reflect.ClassTag

// OpImpl class representing operation implementation, mainly for accessing implementation type information
case class OpImpl[T <: Operation](
    opType: Class[T],
    run: (T, RuntimeCtx) => Unit
)

object OpImpl:

  def apply[T <: Operation: ClassTag](run: (T, RuntimeCtx) => Unit): OpImpl[T] =
    OpImpl(summon[ClassTag[T]].runtimeClass.asInstanceOf[Class[T]], run)

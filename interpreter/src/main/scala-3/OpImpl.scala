package scair.tools

import scair.ir.*
import scala.reflect.ClassTag

case class OpImpl[T <: Operation](opType: Class[T], run: (T, RuntimeCtx) => Unit)

object OpImpl:
  def apply[T <: Operation : ClassTag](run: (T, RuntimeCtx) => Unit): OpImpl[T] =
    OpImpl(summon[ClassTag[T]].runtimeClass.asInstanceOf[Class[T]], run)

// wrap into OpImpl here for readability
inline def summonImplementations(impls: Seq[OpImpl[? <: Operation]]): InterpreterDialect = 
    new InterpreterDialect(impls)



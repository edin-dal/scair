package scair.transformations

import scala.collection.mutable
import scair._

class TransformContext() {

  val passContext: mutable.Map[String, ModulePass] = mutable.Map()

  def getPass(name: String) = passContext.get(name)
}

package scair.transformations

import scala.collection.mutable
import scair._
import scair.utils.allPasses

class TransformContext() {

  val passContext: mutable.Map[String, ModulePass] = mutable.Map()

  for (pass <- allPasses) {
    passContext += pass.name -> pass
  }

  def getPass(name: String) = passContext.get(name)
}

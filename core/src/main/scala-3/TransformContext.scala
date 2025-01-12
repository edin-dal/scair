package scair

import scair.transformations.ModulePass

import scala.collection.mutable

class TransformContext() {

  val passContext: mutable.Map[String, ModulePass] = mutable.Map()

  def getPass(name: String) = passContext.get(name)
}

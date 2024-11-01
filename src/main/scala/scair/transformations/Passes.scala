package scair.transformations

import scair.ir._

abstract class ModulePass {
  val name: String
  def transform(op: Operation): Operation = ???
}

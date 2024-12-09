package scair.transformations

import scair.ir.*

abstract class ModulePass {
  val name: String
  def transform(op: Operation): Operation = ???
}

package scair.transformations

import scair._

abstract class ModulePass {
  val name: String
  def transform(op: Operation): Operation = ???
}

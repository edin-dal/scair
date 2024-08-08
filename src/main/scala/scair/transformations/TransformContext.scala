package scair.transformations

import scair.dialects.CMath.transformations.cdt.{DummyPass}
import scala.collection.mutable
import scair._

private val allPasses: Seq[ModulePass] =
  Seq(DummyPass)

class TransformContext() {

  val passContext: mutable.Map[String, ModulePass] = mutable.Map()

  for (pass <- allPasses) {
    passContext += pass.name -> pass
  }

  def getPass(name: String) = passContext.get(name)
}

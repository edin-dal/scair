package scair

import org.scalatest._
import Matchers._

class MainTest extends FlatSpec {

  "Main" should "test for obvious" in {
    true should be (!false)
  }
}
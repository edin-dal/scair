package scair.verify

import scair.ir.Operation
import scair.utils.OK

trait VerifierCheck:
  def name: String
  def run(root: Operation): OK[Unit]

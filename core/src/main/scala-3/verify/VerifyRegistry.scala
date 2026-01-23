package scair.verify

final class VerifierRegistry:
  private var checks: Vector[VerifierCheck] = Vector.empty
  def register(c: VerifierCheck): Unit = checks :+= c
  def all: Seq[VerifierCheck] = checks

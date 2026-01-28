package scair.verify

final class VerifierRegistry:
  private var checks: Vector[VerifierCheck] = Vector.empty

  def register(c: VerifierCheck): Unit =
    if !checks.exists(_.name == c.name) then checks :+= c

  def registerAll(cs: IterableOnce[VerifierCheck]): Unit =
    cs.iterator.foreach(register)

  def all: Seq[VerifierCheck] = checks

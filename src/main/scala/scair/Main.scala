package scair

case class Region(
    blocks: Seq[Block],
    parent: Option[Operation] = None
) {

  override def toString(): String = {
    return "behold, i am REGION"
  }

  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }
}

case class Block(
    operations: Seq[Operation],
    arguments: Seq[Value]
) {

  override def toString(): String = {
    return "behold, i am block"
  }

  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }
}

case class Attribute(
    name: String
) {

  override def toString(): String = {
    return s"$name"
  }
}

case class Value(
    // val name: String,
    typ: Attribute
) {

  override def toString(): String = {
    return s"%sv"
  }

  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }
}

case class Operation(
    name: String,
    operands: Seq[Value], // TODO rest
    results: Seq[Value],
    regions: Seq[Region]
) {
  override def toString(): String = {
    val resultsStr: String =
      if (results.length > 0) results.mkString(", ") + " = " else ""
    val operandsStr: String =
      if (operands.length > 0) operands.mkString(", ") else ""
    val functionType: String =
      "(" + operands.map(x => x.typ).mkString(", ") + ") -> (" + results
        .map(x => x.typ)
        .mkString(", ") + ")"

    return resultsStr + "\"" + name + "\" (" + operandsStr + ") : " + functionType
  }

  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }
}

object Main {
  def main(args: Array[String]): Unit = {
    println("TODO compiler")
  }
}

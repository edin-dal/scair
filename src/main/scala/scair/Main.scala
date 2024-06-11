package scair

class Region // TODO

case class Block(
	operations: Seq[Operation],
	argumentTypes: Seq[Attribute]
) {

	// Initialization Block
	{
		
	}

	override def toString(): String = {
		return "behold, i am block"
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
	//val name: String,
	typ: Attribute	
){
	override def toString(): String = {
		return s"%sv"
	}
}
	
case class Operation(
	name: String,
	operands: Seq[Value], // TODO rest
	results: Seq[Value]
){
	override def toString(): String = {
		val resultsStr: String = if (results.length > 0) results.mkString(", ") + " = " else ""
		val operandsStr: String = if (operands.length > 0) operands.mkString(", ") else ""
		val functionType: String = "(" + operands.map(x => x.typ).mkString(", ") + ") -> (" + results.map(x => x.typ).mkString(", ") + ")"

		return resultsStr + "\"" + name + "\" (" + operandsStr +") : " + functionType 
	}
}
	
object Main {
	def main(args: Array[String]): Unit = {
		println("TODO compiler")
	}
}
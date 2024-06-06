package scair

class Region // TODO
class Block // TODO

class Attribute(
	val name: String
){
	override def toString(): String = {
		return s"$name"
	}
}
	
class Value(
	val name: String,
	val typ: Attribute	
){
	override def toString(): String = {
		return s"%$name"
	}
}
	
class Operation(
	val name: String,
	val operands: Seq[Value], // TODO rest
	val results: Seq[Value]
){
	override def toString(): String = {
		val resultsStr: String = if (results != None) results.mkString(", ") + " = " else ""
		val operandsStr: String = if (operands != None) operands.mkString(", ") else ""
		val functionType: String = "(" + operands.map(x => x.typ).mkString(", ") + ") -> (" + results.map(x => x.typ).mkString(", ") + ")"

		return resultsStr + "\"" + name + "\" (" + operandsStr +") : (" + functionType + ")" 
		//return s"$resultsStr\"$name\" ($operandsStr) : $functionType)"
	}
}
	
object Main {
	def main(args: Array[String]): Unit = {
		println("TODO compiler")
	}
}
package scair

class Region // TODO
class Block // TODO

class Attribute(
	val name: String
)
	
class Value(
	val typ: Attribute	
)
	
class Operation(
	val name: String,
	val operands: Seq[Value], // TODO rest
	val results: Seq[Value]
)

object Main {
	def main(args: Array[String]): Unit = {
		println("TODO compiler")
	}
}
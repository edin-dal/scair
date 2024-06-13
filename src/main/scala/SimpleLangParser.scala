import fastparse._
import NoWhitespace._

object SimpleLangParser {

  def functionalCall(x: Unit) = {
    println("hello")
  }
  def crazyStuff(x: Int, also: Int): Unit = {
    println("crazyStuff")
  }

  def E[_: P](action: => Unit) = {
    action
    Pass()
  }

  def identifier[_: P]: P[String] = P(
    "{" ~ E(functionalCall()) ~
      CharIn("a-zA-Z_") ~ CharIn("a-zA-Z0-9_").rep ~ "}"
  ).!

  def main(args: Array[String]): Unit = {
    val string = "{aa1111}"

    val result = parse(string, identifier(_))
  }

//   // Basic parsers for identifiers and values
//   def identifier[_: P]: P[String] = P(
//     CharIn("a-zA-Z_") ~ CharIn("a-zA-Z0-9_").rep
//   ).!
//   def value[$: P]: P[String] = P(CharIn("0-9").rep(1).!)

//   // Variable declaration parser
//   def varDeclaration[$: P]: P[(String, String)] =
//     P("var" ~ identifier ~ "=" ~ value ~ ";").map { case (name, v) =>
//       (name, v)
//     }

//   // Parser for a block of code with local scope
//   def block[$: P](implicit scope: Scope): P[Unit] =
//     P("{" ~ statement.rep ~ "}").map(_ => ())

//   // Parser for a statement (either variable declaration or nested block)
//   def statement[$: P](implicit scope: Scope): P[Unit] = P(varDeclaration.map {
//     case (name, v) => scope.declare(name, v)
//   } | block)

//   // Top-level parser
//   def program[$: P](implicit scope: Scope): P[Unit] = P(statement.rep ~ End)

//   // A simple scope class to manage variables
//   class Scope(parent: Option[Scope] = None) {
//     private var variables: Map[String, String] = Map()

//     def declare(name: String, value: String): Unit = {
//       variables += (name -> value)
//       println(s"Declared variable $name with value $value in scope $this")
//     }

//     def lookup(name: String): Option[String] = {
//       variables.get(name).orElse(parent.flatMap(_.lookup(name)))
//     }

//     override def toString: String = s"Scope(${variables.keys.mkString(", ")})"
//   }

//   def main(args: Array[String]): Unit = {
//     val input = "var x = 10;{var y = 20;{var z = 30;}var w = 40;}"

//     implicit val globalScope: Scope = new Scope()

//     val result = parse(input, program(_, globalScope))
//     println(result)
//   }
}

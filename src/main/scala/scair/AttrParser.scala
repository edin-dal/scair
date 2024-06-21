package scair

import fastparse._, MultiLineWhitespace._
import scala.collection.mutable
import scala.util.{Try, Success, Failure}
import IR._

/*

int
float
index

fucntion type
ArrayAttr

StringAttr

DictionaryAttr

memref type

 */

sealed trait Signedness
case object Signed extends Signedness
case object Unsigned extends Signedness
case object Signless extends Signedness

case object Float16Type extends Type("builtin.f16") {
  override def toString = "f16"
}
case object Float32Type extends Type("builtin.f32") {
  override def toString = "f32"
}
case object Float64Type extends Type("builtin.f64") {
  override def toString = "f64"
}
case object Float80Type extends Type("builtin.f80") {
  override def toString = "f80"
}
case object Float128Type extends Type("builtin.f128") {
  override def toString = "f128"
}

case class IntegerType(val width: Int, val sign: Signedness)
    extends Type("builtin.int_type") {
  override def toString = s"i$width"
}

case object IndexType extends Type("builtin.index") {
  override def toString = "index"
}

object AttrParser {

  def Float16TypeP[$: P]: P[Attribute] = P("f16".!).map(_ => Float16Type)
  def Float32TypeP[$: P]: P[Attribute] = P("f32".!).map(_ => Float32Type)
  def Float64TypeP[$: P]: P[Attribute] = P("f64".!).map(_ => Float64Type)
  def Float80TypeP[$: P]: P[Attribute] = P("f80".!).map(_ => Float80Type)
  def Float128TypeP[$: P]: P[Attribute] = P("f128".!).map(_ => Float128Type)

  def IntegerTypeP[$: P]: P[Attribute] =
    P("i" ~~ CharIn("0-9").rep(1).!).map((intString: String) =>
      IntegerType(intString.toInt, Signless)
    )

  def IndexTypeP[$: P]: P[Attribute] = P("index".!).map(_ => IndexType)

  def BuiltIn[$: P]: P[Attribute] = P(
    Float16TypeP | Float32TypeP | Float64TypeP | Float80TypeP | Float128TypeP | IntegerTypeP | IndexTypeP
  )

  def main(args: Array[String]): Unit = {
    val Parsed.Success(x, y) = parse("f16", BuiltIn(_))
    println(x.getClass)
    println("Hello World!")
  }
}

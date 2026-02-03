package scair.interpreter

import scair.ir.*

import scala.collection.mutable

case class ScopedDict(
    val parent: Option[ScopedDict],
    scope: mutable.Map[Value[Attribute], Any],
    name: String
):

  // recursive get function to find variable in current or parent scopes
  def get(key: Value[Attribute]): Option[Any] =
    scope.get(key) match
      case Some(v) => Some(v)
      case None    =>
        parent match
          case Some(p) => p.get(key)
          case None    => None

  def update(key: Value[Attribute], value: Any): Unit =
    scope.update(key, value)

  def pretty_print(): Unit =
    println(s"ScopedDict $name:")
    for (k, v) <- scope do
      println(s"  $k -> $v")

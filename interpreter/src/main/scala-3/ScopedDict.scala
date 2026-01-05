package scair.interpreter

import scala.collection.mutable
import scair.ir.*

case class ScopedDict (
  val parent: Option[ScopedDict],
  scope: mutable.Map[Value[Attribute], Any],
):
    // recursive get function to find variable in current or parent scopes
    def get(key: Value[Attribute]): Option[Any] =
      scope.get(key) match
        case Some(v) => Some(v)
        case None =>
          parent match
            case Some(p) => p.get(key)
            case None    => None

    def update(key: Value[Attribute], value: Any): Unit =
      scope.update(key, value)

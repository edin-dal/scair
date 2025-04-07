package scair.clair.constraint

import scair.ir.Attribute
import scala.quoted._

trait Constraint[Bound <: Attribute] {
    type T = Bound
    def verification(attr : Expr[Bound])(using Quotes, Type[Bound]): Expr[Option[Bound]] =
        '{Some($attr)}
}

class AnyAttr extends Constraint[Attribute] {
}

// Useless by design?
class BaseAttr[T <: Attribute] extends Constraint[T] {
}


class EqualAttr[T <: Attribute](to : T) extends Constraint[T] {
    override def verification(attr: Expr[T])(using Quotes, Type[T]): Expr[Option[T]] =
        '{
            $attr match
                case to => Some($attr)
                case _ => None
        }
}

class AnyOf[T <: Attribute](of : T*) extends Constraint[T]{
    override def verification(attr: Expr[T])(using Quotes, Type[T]): Expr[Option[T]] =
        of match
            case Nil => '{None}
            case h :: t =>
                '{
                    $attr match
                        case h => Some($attr)
                        case or => ${AnyOf(t:_*).verification(attr)}
                }
        
        '{
            $attr match
                case of => Some($attr)
                case or => None
        }

}
trait Constrained[T <: Constraint[_]] extends Attribute {
    // val constraint: Constraint[T]
}
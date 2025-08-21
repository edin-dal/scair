package scair.ir

// ██╗ ██████╗░
// ██║ ██╔══██╗
// ██║ ██████╔╝
// ██║ ██╔══██╗
// ██║ ██║░░██║
// ╚═╝ ╚═╝░░╚═╝

/*≡==--==≡≡≡≡==--=≡≡*\
||      VALUES      ||
\*≡==---==≡≡==---==≡*/

// TO-DO: perhaps a linked list of a use to other uses within an operation
//        for faster use retrieval and index update
case class Use(val operation: Operation, val index: Int)

object Value {
  def apply[T <: Attribute](typ: T): Value[T] = new Value(typ)
  def unapply[T <: Attribute](value: Value[T]): Option[T] = Some(value.typ)
}

class Value[+T <: Attribute](
    val typ: T
) {

  val uses: collection.mutable.Set[Use] = collection.mutable.Set.empty[Use]
  var owner: Option[Operation | Block] = None

  def erase(): Unit = {
    if (uses.nonEmpty) then
      throw new Exception(
        "Attempting to erase a Value that has uses in other operations."
      )
  }

  def verify(): Either[String, Unit] = typ.custom_verify()

  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }

  override def hashCode(): Int = System.identityHashCode(this)

}

extension (seq: Seq[Value[Attribute]]) def typ: Seq[Attribute] = seq.map(_.typ)

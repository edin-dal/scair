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
final case class Use(val operation: Operation, val index: Int)

final case class Value[+T <: Attribute](
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

  override final def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }

  override final def hashCode(): Int = System.identityHashCode(this)

}

type Operand[+T <: Attribute] = Value[T]

opaque type Result[+T <: Attribute] <: Value[T] = Value[T]

object Result:
  def apply[T <: Attribute](typ: T): Result[T] = Value(typ)
  def unapply[T <: Attribute](r: Result[T]): Some[T] = Some(r.typ)
  given [T <: Attribute] => Conversion[Result[T], Value[T]] = identity

extension (seq: Seq[Value[Attribute]]) def typ: Seq[Attribute] = seq.map(_.typ)

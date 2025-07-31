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

  val uses: ListType[Use] = ListType()
  var owner: Option[Operation | Block] = None

  def remove_use(use: Use): Unit = {
    val usesLengthBefore = uses.length
    uses -= use
    if (usesLengthBefore == uses.length) then {
      throw new Exception("Use to be removed was not in the Use list.")
    }
  }

  def erase(): Unit = {
    if (uses.length != 0) then
      throw new Exception(
        "Attempting to erase a Value that has uses in other operations."
      )
  }

  def verify(): Either[String, Unit] = typ.custom_verify()

  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }

}

extension (seq: Seq[Value[Attribute]]) def typ: Seq[Attribute] = seq.map(_.typ)

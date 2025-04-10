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
case class Use(val operation: Operation, val index: Int) {

  override def equals(o: Any): Boolean = o match {
    case Use(op, idx) =>
      (operation eq op) && (index eq idx)
    case _ => super.equals(o)
  }

}

object Value {
  def apply[T <: Attribute](typ: T): Value[T] = new Value(typ)
  def unapply[T <: Attribute](value: Value[T]): Option[T] = Some(value.typ)
}

class Value[+T <: Attribute](
    val typ: T
) {

  var uses: ListType[Use] = ListType()

  def remove_use(use: Use): Unit = {
    val usesLengthBefore = uses.length
    uses -= use
    if (usesLengthBefore == uses.length) then {
      throw new Exception("Use to be removed was not in the Use list.")
    }
  }

  def replace_by(newValue: Value[Attribute]): Unit = {
    for (use <- uses) {
      val op = use.operation
      val block = op.container_block.get
      val new_op =
        op.updated(operands = op.operands.updatedOperandsAndUses(use, newValue))
      val idx = block.operations.indexOf(op)
      block.operations(idx) = new_op
    }
    uses = ListType()
  }

  def erase(): Unit = {
    if (uses.length != 0) then
      throw new Exception(
        "Attempting to erase a Value that has uses in other operations."
      )
  }

  def verify(): Unit = typ.custom_verify()

  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }

}

extension (seq: Seq[Value[Attribute]]) def typ: Seq[Attribute] = seq.map(_.typ)

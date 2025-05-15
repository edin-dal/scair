package scair.ir

import scair.transformations.PatternRewriter
import scair.transformations.RewriteMethods

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

  def remove_use(use: Use): Unit = {
    val usesLengthBefore = uses.length
    uses -= use
    if (usesLengthBefore == uses.length) then {
      throw new Exception("Use to be removed was not in the Use list.")
    }
  }

  def replace_by(newValue: Value[Attribute]): Unit = {
    if !(newValue eq this) then {
      for (use <- Seq.from(uses)) {
        val op = use.operation
        val new_op =
          op.updated(results = op.results, operands = op.operands.updated(use.index, newValue))
        RewriteMethods.replace_op(
          op = op,
          new_ops = new_op,
          new_results = Some(new_op.results)
        )

      }
      uses.clear()
    }
  }

  def erase(): Unit = {
    if (uses.length != 0) then
      throw new Exception(
        "Attempting to erase a Value that has uses in other operations."
      )
  }

  def verify(): Either[Unit, String] = typ.custom_verify()

  override def equals(o: Any): Boolean = {
    return this eq o.asInstanceOf[AnyRef]
  }

}

extension (seq: Seq[Value[Attribute]]) def typ: Seq[Attribute] = seq.map(_.typ)

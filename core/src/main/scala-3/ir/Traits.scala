package scair.ir

// ████████╗ ██████╗░ ░█████╗░ ██╗ ████████╗ ░██████╗
// ╚══██╔══╝ ██╔══██╗ ██╔══██╗ ██║ ╚══██╔══╝ ██╔════╝
// ░░░██║░░░ ██████╔╝ ███████║ ██║ ░░░██║░░░ ╚█████╗░
// ░░░██║░░░ ██╔══██╗ ██╔══██║ ██║ ░░░██║░░░ ░╚═══██╗
// ░░░██║░░░ ██║░░██║ ██║░░██║ ██║ ░░░██║░░░ ██████╔╝
// ░░░╚═╝░░░ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ╚═╝ ░░░╚═╝░░░ ╚═════╝░

/*≡==--=≡≡≡≡=--=≡≡*\
||   TERMINATOR   ||
\*≡==---=≡≡=---==≡*/

trait IsTerminator extends Operation:

  override def trait_verify(): Either[String, Operation] = {
    this.container_block match
      case Some(b) =>
        if this ne b.operations.last then
          Left(
            s"Operation '${name}' marked as a terminator, but is not the last operation within its container block"
          )
        else Right(this)
      case None =>
        Left(
          s"Operation '${name}' marked as a terminator, but is not contained in any block."
        )
  }.flatMap(_ => super.trait_verify())

/*≡==---=≡≡≡≡≡=---=≡≡*\
||   NO TERMINATOR   ||
\*≡==----=≡≡≡=----==≡*/

trait NoTerminator extends Operation:

  override def trait_verify(): Either[String, Operation] = {
    if regions.filter(x => x.blocks.length != 1).length != 0 then
      Left(
        s"NoTerminator Operation '${name}' requires single-block regions"
      )
    else Right(this)
  }.flatMap(_ => super.trait_verify())

trait NoMemoryEffect extends Operation

trait IsolatedFromAbove extends Operation:

  final def verify_rec(regs: Seq[Region]): Either[String, Operation] =
    val r = regs match
      case region :: tail =>
        region.blocks.foldLeft[Either[String, Operation]](Right(this))(
          (r, block) =>
            r.flatMap(_ =>
              block.operations.foldLeft[Either[String, Operation]](r)((r, op) =>
                op.operands
                  .foldLeft(r)((r, o) =>
                    if !this.is_ancestor(
                        o.owner.getOrElse(throw new Exception(s"${op.name}"))
                      )
                    then
                      Left(
                        s"Operation '${name}' is not an ancestor of operand '${o}' of '${op.name}'"
                      )
                    else r
                  )
                  .flatMap(_ => verify_rec(tail ++ op.regions))
              )
            )
        )
      case Nil => Right(this)
    r.flatMap(_ => super.trait_verify())

  override def trait_verify(): Either[String, Operation] =
    verify_rec(regions)

trait Commutative extends Operation

trait ConstantLike(_value: Attribute) extends Operation:
  def getValue: Attribute = _value

object ConstantLike:

  def unapply(op: Operation): Option[(value: Attribute)] =
    op match
      case c: ConstantLike => Some((value = c.getValue))
      case _               => None

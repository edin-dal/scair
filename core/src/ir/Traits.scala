package scair.ir

import scair.dialects.builtin.StringData
import scair.transformations.Rewriter
import scair.utils.*

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

  override def traitVerify(): OK[Operation] = {
    this.containerBlock match
      case Some(b) =>
        if this ne b.operations.last then
          Err(
            s"Operation '$name' marked as a terminator, but is not the last operation within its container block"
          )
        else OK(this)
      case None =>
        Err(
          s"Operation '$name' marked as a terminator, but is not contained in any block."
        )
  }.flatMap(_ => super.traitVerify())

/*≡==---=≡≡≡≡≡=---=≡≡*\
||   NO TERMINATOR   ||
\*≡==----=≡≡≡=----==≡*/

trait NoTerminator extends Operation:

  override def traitVerify(): OK[Operation] = {
    if regions.filter(x => x.blocks.length != 1).length != 0 then
      Err(
        s"NoTerminator Operation '$name' requires single-block regions"
      )
    else OK(this)
  }.flatMap(_ => super.traitVerify())

trait IsolatedFromAbove extends Operation:

  final def verifyRec(regs: Seq[Region]): OK[Operation] =
    val r = regs match
      case region +: tail =>
        region.blocks.foldLeft[OK[Operation]](OK(this))((r, block) =>
          r.flatMap(_ =>
            block.operations.foldLeft[OK[Operation]](r)((r, op) =>
              op.operands.foldLeft(r)((r, o) =>
                if !this
                    .isAncestor(
                      o.owner.getOrElse(throw new Exception(s"${op.name}"))
                    )
                then
                  Err(
                    s"Operation '$name' is not an ancestor of operand '$o' of '${op
                        .name}'"
                  )
                else r
              ).flatMap(_ => verifyRec(tail ++ op.regions))
            )
          )
        )

      case Nil => OK(this)
    r.flatMap(_ => super.traitVerify())

  override def traitVerify(): OK[Operation] =
    verifyRec(regions)

trait Commutative extends Operation

trait ConstantLike(_value: Attribute) extends Operation:
  def getValue: Attribute = _value

object ConstantLike:

  def unapply(op: Operation): Option[(value: Attribute)] =
    op match
      case c: ConstantLike => Some((value = c.getValue))
      case _               => None

/*≡==--==≡≡≡≡≡≡≡≡≡≡≡==--=≡≡*\
||       LOOP LIKE          ||
\*≡==---==≡≡≡≡≡≡≡≡≡==---==≡*/

/** MLIR's `LoopLikeOpInterface`, reduced to what code motion needs. */
trait LoopLike(val loopRegions: Region*) extends Operation:

  /** Whether `value` is defined outside of every loop region.
    *
    * Block arguments of a loop region count as defined inside it, as the parent
    * of their owning block is that region.
    */
  def isDefinedOutsideOfLoop(value: Value[Attribute]): Boolean =
    value.owner match
      case Some(owner) => !loopRegions.exists(_.isAncestor(owner))
      // A detached value is conservatively taken to be inside.
      case None => false

  /** Move `op` out of the loop, to just before it. */
  def moveOutOfLoop(op: Operation)(using rewriter: Rewriter): Unit =
    rewriter.moveOpsBefore(this, op)

object LoopLike:

  def unapply(op: LoopLike): Tuple1[Seq[Region]] = Tuple1(op.loopRegions)

trait Symbol extends Operation:
  def sym_name: StringData

trait SymbolTable extends Operation:
  def regions: Seq[Region]

  override def traitVerify(): OK[Operation] = {
    if regions.length != 1 then
      Err(s"SymbolTable operation '$name' must have exactly one region")
    // else if regions.head.blocks.length != 1 then
    //  Err(s"SymbolTable operation '$name' must have exactly one block in its region")
    else OK(this)
  }.flatMap(_ => super.traitVerify())

object SymbolTable:

  // Look in itself first. Then look in parent symbol tables
  def lookupSymbol(op: SymbolTable, name: String): Option[Operation] =
    op.regions.head.blocks.head.operations.find {
      case s: Symbol if s.sym_name.stringLiteral == name => true
      case _                                             => false
    } match
      case Some(symbol: Symbol) => Some(symbol) // found symbol in current table
      case Some(_) => throw new Exception("Found non-symbol in symbol table")
      case None    =>
        op.containerBlock.flatMap(_.containerRegion)
          .flatMap(_.containerOperation) match
          case Some(symTable: SymbolTable) =>
            lookupSymbol(symTable, name) // look in parent symbol table
          case Some(_) => throw new Exception("Found non-symbol table ancestor")
          case None    =>
            throw new Exception(
              "Reached top-level without finding symbol table ancestor"
            ) // reached top-level without finding symbol table ancestor

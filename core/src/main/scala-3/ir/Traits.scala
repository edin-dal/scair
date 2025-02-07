package scair.ir

// ████████╗ ██████╗░ ░█████╗░ ██╗ ████████╗ ░██████╗
// ╚══██╔══╝ ██╔══██╗ ██╔══██╗ ██║ ╚══██╔══╝ ██╔════╝
// ░░░██║░░░ ██████╔╝ ███████║ ██║ ░░░██║░░░ ╚█████╗░
// ░░░██║░░░ ██╔══██╗ ██╔══██║ ██║ ░░░██║░░░ ░╚═══██╗
// ░░░██║░░░ ██║░░██║ ██║░░██║ ██║ ░░░██║░░░ ██████╔╝
// ░░░╚═╝░░░ ╚═╝░░╚═╝ ╚═╝░░╚═╝ ╚═╝ ░░░╚═╝░░░ ╚═════╝░

/*≡==--=≡≡≡=--=≡≡*\
||    OPTRAIT    ||
\*≡==---=≡=---==≡*/

abstract class OpTrait {
  def op: Operation
  def trait_verify(): Unit = ()
}

/*≡==--=≡≡≡≡=--=≡≡*\
||   TERMINATOR   ||
\*≡==---=≡≡=---==≡*/

trait IsTerminator extends OpTrait {

  abstract override def trait_verify(): Unit = {
    op.container_block match {
      case Some(b) =>
        if (op != b.operations.last) then
          throw new Exception(
            s"Operation '${op.name}' marked as a terminator, but is not the last operation within its container block"
          )
      case None =>
        throw new Exception(
          s"Operation '${op.name}' marked as a terminator, but is not contained in any block."
        )
    }
    super.trait_verify()
  }

}

/*≡==---=≡≡≡≡≡=---=≡≡*\
||   NO TERMINATOR   ||
\*≡==----=≡≡≡=----==≡*/

trait NoTerminator extends OpTrait {

  abstract override def trait_verify(): Unit = {
    if (op.regions.filter(x => x.blocks.length != 1).length != 0) then
      throw new Exception(
        s"NoTerminator Operation '${op.name}' requires single-block regions"
      )

    super.trait_verify()
  }

}

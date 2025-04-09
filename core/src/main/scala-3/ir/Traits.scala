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

trait IsTerminator extends Operation {

  override def trait_verify(): Unit = {
    this.container_block match {
      case Some(b) =>
        if (this != b.operations.last) then
          throw new Exception(
            s"Operation '${name}' marked as a terminator, but is not the last operation within its container block"
          )
      case None =>
        throw new Exception(
          s"Operation '${name}' marked as a terminator, but is not contained in any block."
        )
    }
    super.trait_verify()
  }

}

/*≡==---=≡≡≡≡≡=---=≡≡*\
||   NO TERMINATOR   ||
\*≡==----=≡≡≡=----==≡*/

trait NoTerminator extends Operation {

  override def trait_verify(): Unit = {
    if (regions.filter(x => x.blocks.length != 1).length != 0) then
      throw new Exception(
        s"NoTerminator Operation '${name}' requires single-block regions"
      )

    super.trait_verify()
  }

}

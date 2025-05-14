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

  override def trait_verify(): Either[Operation, String] = {
    {
      this.container_block match {
        case Some(b) =>
          if (this != b.operations.last) then
            Right(
              s"Operation '${name}' marked as a terminator, but is not the last operation within its container block"
            )
          else Left(this)
        case None =>
          Right(
            s"Operation '${name}' marked as a terminator, but is not contained in any block."
          )
      }
    }.orElse(super.trait_verify())
  }

}

/*≡==---=≡≡≡≡≡=---=≡≡*\
||   NO TERMINATOR   ||
\*≡==----=≡≡≡=----==≡*/

trait NoTerminator extends Operation {

  override def trait_verify(): Either[Operation, String] = {
    {
      if (regions.filter(x => x.blocks.length != 1).length != 0) then
        Right(
          s"NoTerminator Operation '${name}' requires single-block regions"
        )
      else Left(this)
    }.orElse(super.trait_verify())
  }

}

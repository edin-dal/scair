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

  override def trait_verify(): Either[String, Operation] = {
    {
      this.container_block match {
        case Some(b) =>
          if (this != b.operations.last) then
            Left(
              s"Operation '${name}' marked as a terminator, but is not the last operation within its container block"
            )
          else Right(this)
        case None => Left(
            s"Operation '${name}' marked as a terminator, but is not contained in any block."
          )
      }
    }.flatMap(_ => super.trait_verify())
  }

}

/*≡==---=≡≡≡≡≡=---=≡≡*\
||   NO TERMINATOR   ||
\*≡==----=≡≡≡=----==≡*/

trait NoTerminator extends Operation {

  override def trait_verify(): Either[String, Operation] = {
    {
      if (regions.filter(x => x.blocks.length != 1).length != 0) then
        Left(s"NoTerminator Operation '${name}' requires single-block regions")
      else Right(this)
    }.flatMap(_ => super.trait_verify())
  }

}

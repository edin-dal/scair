package scair.ir

// ██╗ ██████╗░
// ██║ ██╔══██╗
// ██║ ██████╔╝
// ██║ ██╔══██╗
// ██║ ██║░░██║
// ╚═╝ ╚═╝░░╚═╝

/*≡==--==≡≡≡≡==--=≡≡*\
||     DIALECTS     ||
\*≡==---==≡≡==---==≡*/

final case class Dialect(
    val operations: Seq[MLIROperationObject],
    val attributes: Seq[AttributeObject]
) {}

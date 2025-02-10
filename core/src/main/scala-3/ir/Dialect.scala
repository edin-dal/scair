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
    val operations: Seq[OperationObject],
    val attributes: Seq[AttributeObject]
) {}

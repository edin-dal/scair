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
    val operations: Seq[OperationCompanion],
    val attributes: Seq[AttributeCompanion]
) {}

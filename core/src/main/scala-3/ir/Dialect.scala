//
// ██████╗░ ██╗ ░█████╗░ ██╗░░░░░ ███████╗ ░█████╗░ ████████╗
// ██╔══██╗ ██║ ██╔══██╗ ██║░░░░░ ██╔════╝ ██╔══██╗ ╚══██╔══╝
// ██║░░██║ ██║ ███████║ ██║░░░░░ █████╗░░ ██║░░╚═╝ ░░░██║░░░
// ██║░░██║ ██║ ██╔══██║ ██║░░░░░ ██╔══╝░░ ██║░░██╗ ░░░██║░░░
// ██████╔╝ ██║ ██║░░██║ ███████╗ ███████╗ ╚█████╔╝ ░░░██║░░░
// ╚═════╝░ ╚═╝ ╚═╝░░╚═╝ ╚══════╝ ╚══════╝ ░╚════╝░ ░░░╚═╝░░░
//

package scair.ir

import scair.verify.VerifierCheck

/*≡==--==≡≡≡≡==--=≡≡*\
||     DIALECTS     ||
\*≡==---==≡≡==---==≡*/

final case class Dialect(
    val operations: Seq[OperationCompanion[?]],
    val attributes: Seq[AttributeCompanion[?]],
    val verifierChecks: Seq[VerifierCheck] = Nil,
) {}

package scair.core.utils

case class Args(
    val allow_unregistered: Boolean = false,
    val input: Option[String] = None,
    val skip_verify: Boolean = false,
    val split_input_file: Boolean = false,
    val parsing_diagnostics: Boolean = false,
    val print_generic: Boolean = false,
    val passes: Seq[String] = Seq(),
    val verify_diagnostics: Boolean = false,
    val use_clairV2: Boolean = false    
)

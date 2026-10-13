package scair.tools.opt

import scair.tools.OptBase
//
// ░██████╗ ░█████╗░ ░█████╗░ ██╗ ██████╗░
// ██╔════╝ ██╔══██╗ ██╔══██╗ ██║ ██╔══██╗
// ╚█████╗░ ██║░░╚═╝ ███████║ ██║ ██████╔╝
// ░╚═══██╗ ██║░░██╗ ██╔══██║ ██║ ██╔══██╗
// ██████╔╝ ╚█████╔╝ ██║░░██║ ██║ ██║░░██║
// ╚═════╝░ ░╚════╝░ ╚═╝░░╚═╝ ╚═╝ ╚═╝░░╚═╝
//
// ░█████╗░ ██████╗░ ████████╗
// ██╔══██╗ ██╔══██╗ ╚══██╔══╝
// ██║░░██║ ██████╔╝ ░░░██║░░░
// ██║░░██║ ██╔═══╝░ ░░░██║░░░
// ╚█████╔╝ ██║░░░░░ ░░░██║░░░
// ░╚════╝░ ╚═╝░░░░░ ░░░╚═╝░░░
//

object ScairOpt extends OptBase:
  override def toolName: String = "scair-opt"
  override def dialects = scair.dialects.allDialects
  override def passes = scair.passes.allPasses

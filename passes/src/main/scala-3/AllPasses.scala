package scair.passes

import scair.MLContext
import scair.passes.benchmark_constant_folding.BenchmarkConstantFolding
import scair.passes.canonicalization.Canonicalize
import scair.passes.cdt.DummyPass
import scair.passes.cdt.TestInsertionPass
import scair.passes.cdt.TestReplacementPass
import scair.passes.cse.CommonSubexpressionElimination
import scair.passes.reconcile.ReconcileUnrealizedCasts
import scair.transformations.ModulePass
import scair.passes.MonomorphizePass

//
// ░█████╗░ ██╗░░░░░ ██╗░░░░░
// ██╔══██╗ ██║░░░░░ ██║░░░░░
// ███████║ ██║░░░░░ ██║░░░░░
// ██╔══██║ ██║░░░░░ ██║░░░░░
// ██║░░██║ ███████╗ ███████╗
// ╚═╝░░╚═╝ ╚══════╝ ╚══════╝
//
// ██████╗░ ░█████╗░ ░██████╗ ░██████╗ ███████╗ ░██████╗
// ██╔══██╗ ██╔══██╗ ██╔════╝ ██╔════╝ ██╔════╝ ██╔════╝
// ██████╔╝ ███████║ ╚█████╗░ ╚█████╗░ █████╗░░ ╚█████╗░
// ██╔═══╝░ ██╔══██║ ░╚═══██╗ ░╚═══██╗ ██╔══╝░░ ░╚═══██╗
// ██║░░░░░ ██║░░██║ ██████╔╝ ██████╔╝ ███████╗ ██████╔╝
// ╚═╝░░░░░ ╚═╝░░╚═╝ ╚═════╝░ ╚═════╝░ ╚══════╝ ╚═════╝░
//

val allPasses: Seq[MLContext => ModulePass] =
  Seq(
    BenchmarkConstantFolding(_),
    CommonSubexpressionElimination(_),
    DummyPass(_),
    MonomorphizePass(_),
    ReconcileUnrealizedCasts(_),
    TestInsertionPass(_),
    TestReplacementPass(_),
    Canonicalize(_),
  )

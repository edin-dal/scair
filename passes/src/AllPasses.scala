package scair.passes

import scair.passes.benchmark_constant_folding.BenchmarkConstantFolding
import scair.passes.canonicalization.Canonicalize
import scair.passes.cdt.DummyPass
import scair.passes.cdt.TestInsertionPass
import scair.passes.cdt.TestPassOptions
import scair.passes.cdt.TestReplacementPass
import scair.passes.convert_arith_to_llvm.ConvertArithToLLVM
import scair.passes.convert_func_to_llvm.ConvertFuncToLLVM
import scair.passes.cse.CommonSubexpressionElimination
import scair.passes.licm.LoopInvariantCodeMotion
import scair.passes.reconcile.ReconcileUnrealizedCasts
import scair.transformations.summonPasses

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

val allPasses = summonPasses[
  (
      BenchmarkConstantFolding,
      CommonSubexpressionElimination,
      ConvertArithToLLVM,
      ConvertFuncToLLVM,
      DummyPass,
      ReconcileUnrealizedCasts,
      TestInsertionPass,
      TestReplacementPass,
      TestPassOptions,
      Canonicalize,
      LoopInvariantCodeMotion,
  )
]

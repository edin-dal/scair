// RUN: scair-opt %s | mlir-opt --mlir-print-op-generic | scair-opt | filecheck %s

// The LLVM unit flags round-trip through real mlir-opt. Note that mlir-opt
// parses `llvm.load`'s custom syntax here, so this also pins that syntax to
// MLIR's own, `nontemporal` sitting in the attribute dictionary included.

"func.func"() <{"function_type" = (i32, !llvm.ptr) -> (), "sym_name" = "flags"}> ({
^bb0(%val : i32, %ptr : !llvm.ptr):
  %loaded = "llvm.load"(%ptr) <{"volatile_", "nontemporal", "invariant", "invariantGroup"}> : (!llvm.ptr) -> i32
  %plain = "llvm.load"(%ptr) : (!llvm.ptr) -> i32
  "llvm.store"(%val, %ptr) <{"volatile_", "nontemporal", "invariantGroup"}> : (i32, !llvm.ptr) -> ()
  %sdiv = "llvm.sdiv"(%val, %val) <{"isExact"}> : (i32, i32) -> i32
  %lshr = "llvm.lshr"(%val, %val) <{"isExact"}> : (i32, i32) -> i32
  %ashr = "llvm.ashr"(%val, %val) <{"isExact"}> : (i32, i32) -> i32
  %or = "llvm.or"(%val, %val) <{"isDisjoint"}> : (i32, i32) -> i32
  %zext = "llvm.zext"(%val) <{"nonNeg"}> : (i32) -> i64
  "func.return"() : () -> ()
}) : () -> ()

// CHECK: builtin.module {
// CHECK-NEXT:   func.func @flags(%0: i32, %1: !llvm.ptr) {
// CHECK-NEXT:     %2 = "llvm.load"(%1) <{invariantGroup, volatile_, nontemporal, ordering = 0 : i64, invariant}> : (!llvm.ptr) -> i32
// CHECK-NEXT:     %3 = "llvm.load"(%1) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
// CHECK-NEXT:     "llvm.store"(%0, %1) <{ordering = 0 : i64, volatile_, nontemporal, invariantGroup}> : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:     %4 = "llvm.sdiv"(%0, %0) <{isExact}> : (i32, i32) -> i32
// CHECK-NEXT:     %5 = "llvm.lshr"(%0, %0) <{isExact}> : (i32, i32) -> i32
// CHECK-NEXT:     %6 = "llvm.ashr"(%0, %0) <{isExact}> : (i32, i32) -> i32
// CHECK-NEXT:     %7 = "llvm.or"(%0, %0) <{isDisjoint}> : (i32, i32) -> i32
// CHECK-NEXT:     %8 = "llvm.zext"(%0) <{nonNeg}> : (i32) -> i64
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

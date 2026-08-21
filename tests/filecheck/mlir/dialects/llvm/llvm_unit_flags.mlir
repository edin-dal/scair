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

// CHECK:       func.func @flags(%{{.*}}: i32, %{{.*}}: !llvm.ptr) {
// CHECK-NEXT:    %{{.*}} = llvm.load volatile %{{.*}} invariant invariant_group {nontemporal} : !llvm.ptr -> i32
// CHECK-NEXT:    %{{.*}} = llvm.load %{{.*}} : !llvm.ptr -> i32
// CHECK-NEXT:    "llvm.store"(%{{.*}}, %{{.*}}) <{volatile_, nontemporal, invariantGroup}> : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:    %{{.*}} = "llvm.sdiv"(%{{.*}}, %{{.*}}) <{isExact}> : (i32, i32) -> i32
// CHECK-NEXT:    %{{.*}} = "llvm.lshr"(%{{.*}}, %{{.*}}) <{isExact}> : (i32, i32) -> i32
// CHECK-NEXT:    %{{.*}} = "llvm.ashr"(%{{.*}}, %{{.*}}) <{isExact}> : (i32, i32) -> i32
// CHECK-NEXT:    %{{.*}} = "llvm.or"(%{{.*}}, %{{.*}}) <{isDisjoint}> : (i32, i32) -> i32
// CHECK-NEXT:    %{{.*}} = "llvm.zext"(%{{.*}}) <{nonNeg}> : (i32) -> i64
// CHECK-NEXT:    func.return
// CHECK-NEXT:  }

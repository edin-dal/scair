// RUN: scair-opt %s | filecheck %s
// RUN: scair-opt %s | scair-opt | filecheck %s

// The LLVM unit flags. `llvm.load` uses MLIR's own syntax: keywords, except
// `nontemporal`, which its format leaves to the attribute dictionary;
// `llvm.func` and `llvm.call` carry theirs in their attribute dictionary; the
// rest print generically. The mlir-gated llvm_unit_flags.mlir checks the same
// ops against real mlir-opt.

"llvm.func"() <{"function_type" = () -> (), "sym_name" = "flagged", "no_inline", "dso_local", "arm_streaming"}> ({
  "llvm.return"() : () -> ()
}) : () -> ()

"func.func"() <{"function_type" = (i32, !llvm.ptr) -> (), "sym_name" = "ops"}> ({
^bb0(%val : i32, %ptr : !llvm.ptr):
  %loaded = "llvm.load"(%ptr) <{"volatile_", "nontemporal", "invariant", "invariantGroup"}> : (!llvm.ptr) -> i32
  %plain = "llvm.load"(%ptr) : (!llvm.ptr) -> i32
  "llvm.store"(%val, %ptr) <{"volatile_", "nontemporal", "invariantGroup"}> : (i32, !llvm.ptr) -> ()
  %size = "test.op"() : () -> i32
  %alloca = "llvm.alloca"(%size) <{"elem_type" = i32, "inalloca"}> : (i32) -> !llvm.ptr
  %sdiv = "llvm.sdiv"(%val, %val) <{"isExact"}> : (i32, i32) -> i32
  %lshr = "llvm.lshr"(%val, %val) <{"isExact"}> : (i32, i32) -> i32
  %ashr = "llvm.ashr"(%val, %val) <{"isExact"}> : (i32, i32) -> i32
  %or = "llvm.or"(%val, %val) <{"isDisjoint"}> : (i32, i32) -> i32
  %zext = "llvm.zext"(%val) <{"nonNeg"}> : (i32) -> i64
  "llvm.call"() <{"callee" = @flagged, "convergent", "no_unwind", "will_return", "no_inline", "always_inline", "inline_hint"}> : () -> ()
  %called = "llvm.call"(%val) <{"callee" = @flagged}> : (i32) -> i32
  "llvm.call"() <{"callee" = @flagged, "no_inline"}> {"hello" = "world"} : () -> ()
  "func.return"() : () -> ()
}) : () -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    llvm.func @flagged() attributes {dso_local, no_inline, arm_streaming} {
// CHECK:         }
// CHECK-NEXT:    func.func @ops(%{{.*}}: i32, %{{.*}}: !llvm.ptr) {
// CHECK-NEXT:      %{{.*}} = llvm.load volatile %{{.*}} invariant invariant_group {nontemporal} : !llvm.ptr -> i32
// CHECK-NEXT:      %{{.*}} = llvm.load %{{.*}} : !llvm.ptr -> i32
// CHECK-NEXT:      "llvm.store"(%{{.*}}, %{{.*}}) <{volatile_, nontemporal, invariantGroup}> : (i32, !llvm.ptr) -> ()
// CHECK-NEXT:      %{{.*}} = "test.op"() : () -> i32
// CHECK-NEXT:      %{{.*}} = "llvm.alloca"(%{{.*}}) <{elem_type = i32, inalloca}> : (i32) -> !llvm.ptr
// CHECK-NEXT:      %{{.*}} = "llvm.sdiv"(%{{.*}}, %{{.*}}) <{isExact}> : (i32, i32) -> i32
// CHECK-NEXT:      %{{.*}} = "llvm.lshr"(%{{.*}}, %{{.*}}) <{isExact}> : (i32, i32) -> i32
// CHECK-NEXT:      %{{.*}} = "llvm.ashr"(%{{.*}}, %{{.*}}) <{isExact}> : (i32, i32) -> i32
// CHECK-NEXT:      %{{.*}} = "llvm.or"(%{{.*}}, %{{.*}}) <{isDisjoint}> : (i32, i32) -> i32
// CHECK-NEXT:      %{{.*}} = "llvm.zext"(%{{.*}}) <{nonNeg}> : (i32) -> i64
// CHECK-NEXT:      llvm.call @flagged() {convergent, no_unwind, will_return, no_inline, always_inline, inline_hint} : () -> ()
// CHECK-NEXT:      %{{.*}} = llvm.call @flagged(%{{.*}}) : (i32) -> i32
// CHECK-NEXT:      llvm.call @flagged() {hello = "world", no_inline} : () -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

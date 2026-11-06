// RUN: scair-opt %s | mlir-opt --allow-unregistered-dialect --mlir-print-op-generic | scair-opt | filecheck %s

"memref.global"() <{sym_name = "x", sym_visibility = "private", type = memref<2xf32>, initial_value = dense<[0.0, 2.0]> : tensor<2xf32>}> : () -> ()
"memref.global"() <{sym_name = "y", sym_visibility = "private", type = memref<4xi32>}> : () -> ()
"memref.global"() <{sym_name = "z", type = memref<3xf16>, initial_value}> : () -> ()
"memref.global"() <{sym_name = "c", type = memref<2xi32>, initial_value = dense<[1, 4]> : tensor<2xi32>, constant}> : () -> ()
"memref.global"() <{sym_name = "a", sym_visibility = "private", type = memref<2xf32>, initial_value = dense<[0.0, 2.0]> : tensor<2xf32>, alignment = 64 : i64}> : () -> ()
"memref.global"() <{sym_name = "s", sym_visibility = "public", type = memref<f32, 3>, initial_value = dense<1.0> : tensor<f32>}> : () -> ()

%g0 = "memref.get_global"() <{name = @x}> : () -> memref<2xf32>
%g1 = "memref.get_global"() <{name = @c}> : () -> memref<2xi32>
%g2 = "memref.get_global"() <{name = @s}> : () -> memref<f32, 3>

// CHECK:       builtin.module {
// CHECK-NEXT:    "memref.global"() <{sym_name = "x", type = memref<2xf32>, sym_visibility = "private", initial_value = dense<[0.0, 2.0]> : tensor<2xf32>}> : () -> ()
// CHECK-NEXT:    "memref.global"() <{sym_name = "y", type = memref<4xi32>, sym_visibility = "private"}> : () -> ()
// CHECK-NEXT:    "memref.global"() <{sym_name = "z", type = memref<3xf16>, initial_value}> : () -> ()
// CHECK-NEXT:    "memref.global"() <{sym_name = "c", type = memref<2xi32>, initial_value = dense<[1, 4]> : tensor<2xi32>, constant}> : () -> ()
// CHECK-NEXT:    "memref.global"() <{sym_name = "a", alignment = 64 : i64, type = memref<2xf32>, initial_value = dense<[0.0, 2.0]> : tensor<2xf32>, sym_visibility = "private"}> : () -> ()
// CHECK-NEXT:    "memref.global"() <{sym_name = "s", type = memref<f32, 3 : i64>, sym_visibility = "public", initial_value = dense<1.0> : tensor<f32>}> : () -> ()
// CHECK-NEXT:    %0 = memref.get_global @x : memref<2xf32>
// CHECK-NEXT:    %1 = memref.get_global @c : memref<2xi32>
// CHECK-NEXT:    %2 = memref.get_global @s : memref<f32, 3 : i64>
// CHECK-NEXT:  }

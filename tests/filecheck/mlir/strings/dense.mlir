// Dense elements hold strings, or raw data written as a hexadecimal string.

// RUN: mlir-opt %s --allow-unregistered-dialect --mlir-print-op-generic | filecheck %s --strict-whitespace
// RUN: scair-opt %s --allow-unregistered-dialect --print-generic | filecheck %s --strict-whitespace

// CHECK: "builtin.module"() ({

// String elements.
"test.op"() {s = dense<"\FF"> : tensor<!foo.str>} : () -> ()
// CHECK-NEXT:   "test.op"() {s = dense<"\FF"> : tensor<!foo.str>} : () -> ()

"test.op"() {s = dense<["a", "\FF"]> : tensor<2x!foo.str>} : () -> ()
// CHECK-NEXT:   "test.op"() {s = dense<["a", "\FF"]> : tensor<2x!foo.str>} : () -> ()

"test.op"() {s = dense<["a\nb", "\22"]> : tensor<2x!foo.str>} : () -> ()
// CHECK-NEXT:   "test.op"() {s = dense<["a\0Ab", "\22"]> : tensor<2x!foo.str>} : () -> ()

// Raw data in hexadecimal, in little-endian order.
"test.op"() {s = dense<"0xDEADBEEF"> : tensor<2xi16>} : () -> ()
// CHECK-NEXT:   "test.op"() {s = dense<[-21026, -4162]> : tensor<2xi16>} : () -> ()

"test.op"() {s = dense<"0xdeadbeef"> : tensor<4xi8>} : () -> ()
// CHECK-NEXT:   "test.op"() {s = dense<[-34, -83, -66, -17]> : tensor<4xi8>} : () -> ()

// Above 100 elements, raw data prints in hexadecimal.
"test.op"() {s = dense<"0x000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F202122232425262728292A2B2C2D2E2F303132333435363738393A3B3C3D3E3F404142434445464748494A4B4C4D4E4F505152535455565758595A5B5C5D5E5F606162636465666768696A6B6C6D6E6F707172737475767778797A7B7C7D7E7F808182838485868788898A8B8C8D8E8F909192939495969798999A9B9C9D9E9F"> : tensor<160xi8>} : () -> ()
// CHECK-NEXT:   "test.op"() {s = dense<"0x000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F202122232425262728292A2B2C2D2E2F303132333435363738393A3B3C3D3E3F404142434445464748494A4B4C4D4E4F505152535455565758595A5B5C5D5E5F606162636465666768696A6B6C6D6E6F707172737475767778797A7B7C7D7E7F808182838485868788898A8B8C8D8E8F909192939495969798999A9B9C9D9E9F"> : tensor<160xi8>} : () -> ()

// CHECK-NEXT: }) : () -> ()

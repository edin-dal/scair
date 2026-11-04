// Strings are not text: UTF-8 is only one of the byte sequences they can hold.

// RUN: mlir-opt %s --allow-unregistered-dialect --mlir-print-op-generic | filecheck %s --strict-whitespace
// RUN: scair-opt %s --allow-unregistered-dialect --print-generic | filecheck %s --strict-whitespace
// RUN: scair-opt %s --allow-unregistered-dialect --print-generic -p 'mlir-opt[]' | filecheck %s --strict-whitespace

// CHECK: "builtin.module"() ({

// Raw UTF-8 in the source denotes its bytes, which print as escapes.
"test.op"() {s = "é"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\C3\A9"} : () -> ()

"test.op"() {s = "ÿ"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\C3\BF"} : () -> ()

"test.op"() {s = "€"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\E2\82\AC"} : () -> ()

"test.op"() {s = "😀"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\F0\9F\98\80"} : () -> ()

"test.op"() {s = "日本語"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\E6\97\A5\E6\9C\AC\E8\AA\9E"} : () -> ()

"test.op"() {s = "aé b€ c😀"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\C3\A9 b\E2\82\AC c\F0\9F\98\80"} : () -> ()

"test.op"() {s = ""} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\C2\80"} : () -> ()

"test.op"() {s = "߿"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\DF\BF"} : () -> ()

"test.op"() {s = "ࠀ"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\E0\A0\80"} : () -> ()

"test.op"() {s = "￿"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\EF\BF\BF"} : () -> ()

"test.op"() {s = "𐀀"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\F0\90\80\80"} : () -> ()

"test.op"() {s = "􏿿"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\F4\8F\BF\BF"} : () -> ()

"test.op"() {s = "�"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\EF\BF\BD"} : () -> ()

"test.op"() {s = "﻿"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\EF\BB\BF"} : () -> ()

"test.op"() {s = " "} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\E2\80\A8"} : () -> ()

"test.op"() {s = " "} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\E2\80\A9"} : () -> ()

"test.op"() {s = " "} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\C2\A0"} : () -> ()

"test.op"() {s = "é"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "e\CC\81"} : () -> ()

// Raw UTF-8 and its escaped bytes are the same string.
"test.op"() {s = "\C3\A9"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\C3\A9"} : () -> ()

"test.op"() {s = "\E2\82\AC"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\E2\82\AC"} : () -> ()

"test.op"() {s = "\F0\9F\98\80"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\F0\9F\98\80"} : () -> ()

"test.op"() {s = "é\C3\A9é"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\C3\A9\C3\A9\C3\A9"} : () -> ()

"test.op"() {s = "\C3\A9é\C3\A9"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\C3\A9\C3\A9\C3\A9"} : () -> ()

// A byte that is not ASCII is not the Unicode code point of the same value.
"test.op"() {s = "\E9"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\E9"} : () -> ()

"test.op"() {s = "\FF"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\FF"} : () -> ()

"test.op"() {s = "\80"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\80"} : () -> ()

"test.op"() {s = "\C3"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\C3"} : () -> ()

"test.op"() {s = "\A9"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\A9"} : () -> ()

"test.op"() {s = "ÿ\FF"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\C3\BF\FF"} : () -> ()

"test.op"() {s = "é\E9"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\C3\A9\E9"} : () -> ()

// Ill-formed UTF-8 written as escapes: stray continuation, truncated, overlong.
"test.op"() {s = "\80"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\80"} : () -> ()

"test.op"() {s = "\BF"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\BF"} : () -> ()

"test.op"() {s = "\80\80"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\80\80"} : () -> ()

"test.op"() {s = "\C3"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\C3"} : () -> ()

"test.op"() {s = "\E2\82"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\E2\82"} : () -> ()

"test.op"() {s = "\F0\9F\98"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\F0\9F\98"} : () -> ()

"test.op"() {s = "\C0\80"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\C0\80"} : () -> ()

"test.op"() {s = "\C1\BF"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\C1\BF"} : () -> ()

"test.op"() {s = "\E0\80\80"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\E0\80\80"} : () -> ()

"test.op"() {s = "\F0\80\80\80"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\F0\80\80\80"} : () -> ()

"test.op"() {s = "a\C3b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\C3b"} : () -> ()

"test.op"() {s = "a\E2\82b"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "a\E2\82b"} : () -> ()

"test.op"() {s = "\C3\C3\A9"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\C3\C3\A9"} : () -> ()

// Ill-formed UTF-8 written as escapes: surrogates, out of range, invalid lead bytes.
"test.op"() {s = "\ED\A0\80"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\ED\A0\80"} : () -> ()

"test.op"() {s = "\ED\BF\BF"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\ED\BF\BF"} : () -> ()

"test.op"() {s = "\ED\A0\BD\ED\B8\80"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\ED\A0\BD\ED\B8\80"} : () -> ()

"test.op"() {s = "\F4\90\80\80"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\F4\90\80\80"} : () -> ()

"test.op"() {s = "\F5\80\80\80"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\F5\80\80\80"} : () -> ()

"test.op"() {s = "\F8\88\80\80\80"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\F8\88\80\80\80"} : () -> ()

"test.op"() {s = "\FC\84\80\80\80\80"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\FC\84\80\80\80\80"} : () -> ()

"test.op"() {s = "\FE"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\FE"} : () -> ()

"test.op"() {s = "\FF"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\FF"} : () -> ()

"test.op"() {s = "\FE\FF"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\FE\FF"} : () -> ()

"test.op"() {s = "\FF\FE"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\FF\FE"} : () -> ()

// Encoded U+FFFD is distinct from the bytes a lossy decoder would replace with it.
"test.op"() {s = "\EF\BF\BD"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\EF\BF\BD"} : () -> ()

"test.op"() {s = "\FF\EF\BF\BD\FF"} : () -> ()
// CHECK-NEXT:   "test.op"() {s = "\FF\EF\BF\BD\FF"} : () -> ()

// CHECK-NEXT: }) : () -> ()

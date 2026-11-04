// A string cut short by the end of the input, at any point of an escape, is an
// error and not a crash.

// RUN: printf '%%s' '"test.op"() {s = "' | mlir-opt --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: printf '%%s' '"test.op"() {s = "' | scair-opt --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: printf '%%s' '"test.op"() {s = "abc' | mlir-opt --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: printf '%%s' '"test.op"() {s = "abc' | scair-opt --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: printf '%%s' '"test.op"() {s = "abc\' | mlir-opt --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: printf '%%s' '"test.op"() {s = "abc\' | scair-opt --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: printf '%%s' '"test.op"() {s = "abc\4' | mlir-opt --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: printf '%%s' '"test.op"() {s = "abc\4' | scair-opt --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: printf '%%s' '"test.op"() {s = "abc\41' | mlir-opt --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: printf '%%s' '"test.op"() {s = "abc\41' | scair-opt --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: printf '%%s' '"test.op"() {s = "abc\"' | mlir-opt --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: printf '%%s' '"test.op"() {s = "abc\"' | scair-opt --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: printf '%%s' '"test.op"() {s = "abc\n' | mlir-opt --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: printf '%%s' '"test.op"() {s = "abc\n' | scair-opt --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: printf '%%s' '"test.op"() {s = "abc\\' | mlir-opt --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: printf '%%s' '"test.op"() {s = "abc\\' | scair-opt --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: printf '%%s' '"test.op"() {"abc\' | mlir-opt --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: printf '%%s' '"test.op"() {"abc\' | scair-opt --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: printf '%%s' '"test.op"() {s = @"abc\' | mlir-opt --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: printf '%%s' '"test.op"() {s = @"abc\' | scair-opt --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: printf '%%s' '"foo\' | mlir-opt --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: printf '%%s' '"foo\' | scair-opt --allow-unregistered-dialect 2>&1 | filecheck %s

// Line breaks that end a string: vertical tab, form feed, and CRLF.
// RUN: printf '%b' '"test.op"() {s = "a\x0Bb"} : () -> ()\n' | mlir-opt --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: printf '%b' '"test.op"() {s = "a\x0Bb"} : () -> ()\n' | scair-opt --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: printf '%b' '"test.op"() {s = "a\x0Cb"} : () -> ()\n' | mlir-opt --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: printf '%b' '"test.op"() {s = "a\x0Cb"} : () -> ()\n' | scair-opt --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: printf '%b' '"test.op"() {s = "a\r\nb"} : () -> ()\n' | mlir-opt --allow-unregistered-dialect 2>&1 | filecheck %s
// RUN: printf '%b' '"test.op"() {s = "a\r\nb"} : () -> ()\n' | scair-opt --allow-unregistered-dialect 2>&1 | filecheck %s

// CHECK-NOT: Exception
// CHECK: {{: error: |^Parse error at }}

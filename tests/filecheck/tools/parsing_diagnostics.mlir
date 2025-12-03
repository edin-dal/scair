// RUN: scair-opt --split-input-file --parsing-diagnostics %s | filecheck %s -DFILE=%s


notir

// CHECK:       Parse error at [[FILE]]:4:1:

// CHECK:       notir
// CHECK-NEXT:  ^
// CHECK-NEXT:  end-of-input

// -----
notir

// CHECK:       Parse error at [[FILE]]:13:1:

// CHECK:       notir
// CHECK-NEXT:  ^
// CHECK-NEXT:  end-of-input

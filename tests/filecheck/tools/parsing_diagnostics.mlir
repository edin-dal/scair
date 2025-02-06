// RUN: scair-opt --split_input_file --parsing_diagnostics %s | filecheck %s -DFILE=%s


notir

// CHECK:       Parse error at [[FILE]]:4:1:

// CHECK:       notir
// CHECK-NEXT:  ^
// CHECK-NEXT:  end-of-input

// -----
notir

// CHECK:       Parse error at [[FILE]]:1:1:

// CHECK:       notir
// CHECK-NEXT:  ^
// CHECK-NEXT:  end-of-input

// Symbols through the custom syntax of the func dialect.

// RUN: mlir-opt %s | filecheck %s --strict-whitespace
// RUN: scair-opt %s | filecheck %s --strict-whitespace

// CHECK: {{(builtin\.)?}}module {

func.func private @"plain"()
// CHECK-NEXT:   func.func private @plain()

func.func private @"\70lain2"()
// CHECK-NEXT:   func.func private @plain2()

func.func private @"a b"()
// CHECK-NEXT:   func.func private @"a b"()

func.func private @"a\20c"()
// CHECK-NEXT:   func.func private @"a c"()

func.func private @"a-b"()
// CHECK-NEXT:   func.func private @"a-b"()

func.func private @"0a"()
// CHECK-NEXT:   func.func private @"0a"()

func.func private @"\22"()
// CHECK-NEXT:   func.func private @"\22"()

func.func private @"\5C"()
// CHECK-NEXT:   func.func private @"\\"()

func.func private @"\0A"()
// CHECK-NEXT:   func.func private @"\0A"()

func.func private @"\00"()
// CHECK-NEXT:   func.func private @"\00"()

func.func private @"\FF"()
// CHECK-NEXT:   func.func private @"\FF"()

func.func private @"é"()
// CHECK-NEXT:   func.func private @"\C3\A9"()

// Calls resolve to the decoded symbol. ScaIR prints them in the generic syntax, which is unrelated.
func.func @caller() {
  "func.call"() <{callee = @"a\20b"}> : () -> ()
  "func.call"() <{callee = @"\FF"}> : () -> ()
  "func.call"() <{callee = @"\22"}> : () -> ()
  "func.return"() : () -> ()
}
// CHECK-NEXT:   func.func @caller() {
// CHECK-NEXT:     {{.*call.*}}@"a b"
// CHECK-NEXT:     {{.*call.*}}@"\FF"
// CHECK-NEXT:     {{.*call.*}}@"\22"
// CHECK-NEXT:     {{(func\.)?}}return
// CHECK-NEXT:   }

// CHECK-NEXT: }

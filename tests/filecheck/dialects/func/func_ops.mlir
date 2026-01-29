// RUN: scair-opt %s | filecheck %s

func.func @noarg_void() {
  func.return
}

func.func @noarg_attributes() attributes {hello = "world"} {
  func.return
}

func.func @call_void() {
  "func.call"() <{"callee" = @call_void}> : () -> ()
  func.return
}

func.func @call_void_attributes() {
  "func.call"() <{"callee" = @call_void_attributes}> {"hello" = "world"} : () -> ()
  func.return
}

func.func @arg_rec(%0 : index) -> index {
   %1 = "func.call"(%0) <{"callee" = @arg_rec}> : (index) -> index
  func.return %1 : index
}

func.func @arg_rec_block(%0 : index) -> index {
  %1 = "func.call"(%0) <{"callee" = @arg_rec_block}> : (index) -> index
  func.return %1 : index
}

func.func private @external_fn(i32) -> (i32, i32)

func.func @multi_return_body(%0 : i32) -> (i32, i32) {
  func.return %0, %0 : i32, i32
}

func.func @constant_to_void() {
  %0 = func.constant @noarg_void : () -> ()
  func.return
}

func.func @constant_to_external() {
  %0 = func.constant @external_fn : (i32) -> (i32, i32)
  func.return
}

func.func @call_indirect_void() {
  %0 = func.constant @noarg_void : () -> ()
  "func.call_indirect"(%0) : (() -> ()) -> ()
  func.return
}

func.func @call_indirect_external(%0 : i32) -> (i32, i32) {
  %1 = func.constant @external_fn : (i32) -> (i32, i32)
  %2, %3 = "func.call_indirect"(%1, %0) : ((i32) -> (i32, i32), i32) -> (i32, i32)
  func.return %2, %3 : i32, i32
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @noarg_void() {
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @noarg_attributes() attributes {hello = "world"} {
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @call_void() {
// CHECK-NEXT:      "func.call"() <{callee = @call_void}> : () -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @call_void_attributes() {
// CHECK-NEXT:      "func.call"() <{callee = @call_void_attributes}> {hello = "world"} : () -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @arg_rec(%0: index) -> index {
// CHECK-NEXT:      %1 = "func.call"(%0) <{callee = @arg_rec}> : (index) -> index
// CHECK-NEXT:      func.return %1 : index
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @arg_rec_block(%0: index) -> index {
// CHECK-NEXT:      %1 = "func.call"(%0) <{callee = @arg_rec_block}> : (index) -> index
// CHECK-NEXT:      func.return %1 : index
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func private @external_fn(i32) -> (i32, i32)
// CHECK-NEXT:    func.func @multi_return_body(%0: i32) -> (i32, i32) {
// CHECK-NEXT:      func.return %0, %0 : i32, i32
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @constant_to_void() {
// CHECK-NEXT:      %0 = func.constant @noarg_void : () -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @constant_to_external() {
// CHECK-NEXT:      %0 = func.constant @external_fn : (i32) -> (i32, i32)
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @call_indirect_void() {
// CHECK-NEXT:      %0 = func.constant @noarg_void : () -> ()
// CHECK-NEXT:      "func.call_indirect"(%0) : (() -> ()) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @call_indirect_external(%0: i32) -> (i32, i32) {
// CHECK-NEXT:      %1 = func.constant @external_fn : (i32) -> (i32, i32)
// CHECK-NEXT:      %2, %3 = "func.call_indirect"(%1, %0) : ((i32) -> (i32, i32), i32) -> (i32, i32)
// CHECK-NEXT:      func.return %2, %3 : i32, i32
// CHECK-NEXT:    }
// CHECK-NEXT:  }

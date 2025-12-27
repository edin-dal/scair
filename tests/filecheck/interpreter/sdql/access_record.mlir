// RUN: scair-run %s | filecheck %s

builtin.module {
  func.func @main() -> i32 {
    %0 = "arith.constant"() <{value = 15 : i32}> : () -> i32

    %1 = "arith.constant"() <{value = 10 : i32}> : () -> i32

    %rec = sdql.create_record {fields = ["a", "b"]} %0, %1 : i32, i32 -> record<"a": i32, "b": i32>

    %val = sdql.access_record %rec "a" : record<"a": i32, "b": i32> -> i32

    func.return %val : i32
  }
}


// CHECK: 15
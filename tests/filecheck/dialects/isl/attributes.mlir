// RUN: scair-opt %s | filecheck %s

builtin.module {
  "test.op"() {map = #isl.map<{ [i] -> [j] :  }>} : () -> ()
  "test.op"() {map = #isl.map<{ [i] -> [j] : i >= 0 and j >= 0 }>} : () -> ()
  "test.op"() {map = #isl.map<{ [i, j] -> [0] : i > j }>} : () -> ()
  "test.op"() {map = #isl.map<[N, M] -> { [i, j] -> [0] : i > j and i < N and j < M }>} : () -> ()
  "test.op"() {map = #isl.set<{ [i, j] : i >= 0 and j >= 0 and i + j <= 10 }>} : () -> ()
}

// CHECK:       builtin.module {
// CHECK-NEXT:    "test.op"() {map = #isl.map<{ [i] -> [j] }>} : () -> ()
// CHECK-NEXT:    "test.op"() {map = #isl.map<{ [i] -> [j] : i >= 0 and j >= 0 }>} : () -> ()
// CHECK-NEXT:    "test.op"() {map = #isl.map<{ [i, j] -> [0] : j < i }>} : () -> ()
// CHECK-NEXT:    "test.op"() {map = #isl.map<[N, M] -> { [i, j] -> [0] : i < N and j < i and j < M }>} : () -> ()
// CHECK-NEXT:    "test.op"() {map = #isl.set<{ [i, j] : i >= 0 and 0 <= j <= 10 - i }>} : () -> ()
// CHECK-NEXT:  }

// RUN: scair-opt %s --allow-unregistered-dialect | filecheck %s
// RUN: scair-opt %s --allow-unregistered-dialect | scair-opt --allow-unregistered-dialect | filecheck %s

func.func @number_attr_f64() {
  "test.number_attr"() {
    // CHECK: attr = #complex.number<:f64 1.0, 0.0> : complex<f64>
    attr = #complex.number<:f64 1.0, 0.0>
  } : () -> ()

  func.return
}

func.func @number_attr_f32() {
  "test.number_attr"() {
    // CHECK: attr = #complex.number<:f32 1.0, 0.0> : complex<f32>
    attr = #complex.number<:f32 1.0, 0.0>
  } : () -> ()

  func.return
}
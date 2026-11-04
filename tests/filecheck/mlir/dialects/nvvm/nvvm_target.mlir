// RUN: scair-opt %s | scair-opt | filecheck %s

// Positional form: mlir-opt prints and parses #nvvm.target with keywords.

"test.op"() {target = #nvvm.target<2 : i32, "nvptx64-nvidia-cuda", "sm_50", "", {}, [], true>} : () -> ()
"test.op"() {target = #nvvm.target<3 : i32, "nvptx64-nvidia-cuda", "sm_90", "+ptx80", {fast}, ["libdevice.bc", "extra.bc"], false>} : () -> ()

// CHECK: "test.op"() {target = #nvvm.target<2 : i32, "nvptx64-nvidia-cuda", "sm_50", "",  {}, [], true>} : () -> ()
// CHECK: "test.op"() {target = #nvvm.target<3 : i32, "nvptx64-nvidia-cuda", "sm_90", "+ptx80",  {fast}, ["libdevice.bc", "extra.bc"], false>} : () -> ()

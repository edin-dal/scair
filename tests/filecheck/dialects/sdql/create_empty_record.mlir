// RUN: scair-opt %s | filecheck %s

%res = sdql.create_record {fields = []} -> record<>

// CHECK: builtin.module {
// CHECK:   %0 = sdql.create_record {fields = []} -> record<>
// CHECK: }

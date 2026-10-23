// The test pass records its non-default options in `test.options`. That
// attribute is only visible in the generic format, so every run prints with -g.

// RUN: scair-opt -g %s -p test-pass-options | filecheck %s --check-prefix=DEFAULT
// RUN: scair-opt -g %s -p 'test-pass-options{number=3 ratio=2.5 label="a b, {c}" flag maybe=7 sizes=1,2,3 order=bottom-up}' | filecheck %s --check-prefix=ALL
// RUN: scair-opt -g %s -p 'test-pass-options{flag=false number=1}' | filecheck %s --check-prefix=DEFAULT
// RUN: scair-opt -g %s -p 'test-pass-options{number=2} , test-pass-options{sizes=4}' | filecheck %s --check-prefix=TWICE

builtin.module {}

// DEFAULT:      "builtin.module"() ({
// DEFAULT-NEXT: }) {test.options = "test-pass-options"} : () -> ()

// ALL:      "builtin.module"() ({
// ALL-NEXT: }) {test.options = "test-pass-options{number=3 ratio=2.5 label=\"a b, {c}\" flag=true maybe=7 sizes=1,2,3 order=bottom-up}"} : () -> ()

// The last pass in the pipeline overwrites the attribute set by the earlier one.
// TWICE:      "builtin.module"() ({
// TWICE-NEXT: }) {test.options = "test-pass-options{sizes=4}"} : () -> ()

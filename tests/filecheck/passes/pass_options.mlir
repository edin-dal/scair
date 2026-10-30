// The test pass records each parsed field as an attribute. Every run prints
// with -g so those attributes are visible.

// RUN: scair-opt -g %s -p TestPassOptions | filecheck %s --check-prefix=DEFAULT
// RUN: scair-opt -g %s -p 'TestPassOptions{number=3 ratio=2.5 label="a b, {c}" flag maybe=7 sizes=1,2,3 order=BottomUp}' | filecheck %s --check-prefix=ALL
// RUN: scair-opt -g %s -p 'TestPassOptions{flag=false number=1}' | filecheck %s --check-prefix=DEFAULT
// RUN: scair-opt -g %s -p 'TestPassOptions{number=2} , TestPassOptions{sizes=4}' | filecheck %s --check-prefix=TWICE
// RUN: scair-opt -g %s -p 'TestPassOptions{flag}' | filecheck %s --check-prefix=BARE

builtin.module {}

// DEFAULT-DAG: test.number = 1
// DEFAULT-DAG: test.ratio = 0.5
// DEFAULT-DAG: test.label = "default"
// DEFAULT-DAG: test.flag = "false"
// DEFAULT-DAG: test.maybe = "None"
// DEFAULT-DAG: test.sizes = []
// DEFAULT-DAG: test.order = "TopDown"

// ALL-DAG: test.number = 3
// ALL-DAG: test.ratio = 2.5
// ALL-DAG: test.label = "a b, {c}"
// ALL-DAG: test.flag = "true"
// ALL-DAG: test.maybe = "Some(7)"
// ALL-DAG: test.sizes = [1, 2, 3]
// ALL-DAG: test.order = "BottomUp"

// The last pass overwrites the attributes set by the earlier one.
// TWICE-DAG: test.number = 1
// TWICE-DAG: test.sizes = [4]

// BARE-DAG: test.flag = "true"

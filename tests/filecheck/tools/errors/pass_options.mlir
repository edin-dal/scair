// RUN: scair-opt %s -p 'TestPassOptions{nubmer=3}' 2>&1 | filecheck %s --check-prefix=UNKNOWN
// RUN: scair-opt %s -p 'TestPassOptions{number=three}' 2>&1 | filecheck %s --check-prefix=BADVALUE
// RUN: scair-opt %s -p 'TestPassOptions{order=sideways}' 2>&1 | filecheck %s --check-prefix=BADENUM
// RUN: scair-opt %s -p 'TestPassOptions{number=3' 2>&1 | filecheck %s --check-prefix=MALFORMED
// RUN: scair-opt %s -p 'TestPassOptions{number=1 number=2}' 2>&1 | filecheck %s --check-prefix=DUPLICATE
// RUN: scair-opt %s -p 'not-a-pass{x=1}' 2>&1 | filecheck %s --check-prefix=NOPASS

builtin.module {}

// UNKNOWN-NOT: Parse error at
// UNKNOWN: {{^}}TestPassOptions{nubmer=3}
// UNKNOWN-NEXT: ^
// UNKNOWN-NEXT: TestPassOptions: unknown option 'nubmer', expected one of: number, ratio, label, flag, maybe, sizes, order

// BADVALUE: {{^}}TestPassOptions{number=three}
// BADVALUE-NEXT: ^
// BADVALUE-NEXT: TestPassOptions: option 'number': expected Int

// BADENUM: {{^}}TestPassOptions{order=sideways}
// BADENUM-NEXT: ^
// BADENUM-NEXT: TestPassOptions: option 'order': expected TopDown|BottomUp

// MALFORMED: {{^}}TestPassOptions{number=3
// MALFORMED-NEXT: ^
// MALFORMED-NEXT: "}"

// NOPASS:      {{^}}not-a-pass{x=1}
// NOPASS-NEXT: ^
// NOPASS-NEXT: 'not-a-pass' does not refer to a registered pass.
// NOPASS-NEXT: Currently registered passes are:
// NOPASS:        - TestPassOptions{{$}}

// DUPLICATE: {{^}}TestPassOptions{number=1 number=2}
// DUPLICATE-NEXT: ^
// DUPLICATE-NEXT: TestPassOptions: duplicate option 'number'

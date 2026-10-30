// RUN: scair-opt %s -p 'test-pass-options{nubmer=3}' 2>&1 | filecheck %s --check-prefix=UNKNOWN
// RUN: scair-opt %s -p 'test-pass-options{number=three}' 2>&1 | filecheck %s --check-prefix=BADVALUE
// RUN: scair-opt %s -p 'test-pass-options{order=sideways}' 2>&1 | filecheck %s --check-prefix=BADENUM
// RUN: scair-opt %s -p 'test-pass-options{number=3' 2>&1 | filecheck %s --check-prefix=MALFORMED
// RUN: scair-opt %s -p 'test-pass-options{number=1 number=2}' 2>&1 | filecheck %s --check-prefix=DUPLICATE
// RUN: scair-opt %s -p 'not-a-pass{x=1}' 2>&1 | filecheck %s --check-prefix=NOPASS

builtin.module {}

// UNKNOWN-NOT: Parse error at
// UNKNOWN: {{^}}test-pass-options{nubmer=3}
// UNKNOWN-NEXT: ^
// UNKNOWN-NEXT: test-pass-options: unknown option 'nubmer', expected one of: number, ratio, label, flag, maybe, sizes, order

// BADVALUE: {{^}}test-pass-options{number=three}
// BADVALUE-NEXT: ^
// BADVALUE-NEXT: test-pass-options: option 'number': expected Int

// BADENUM: {{^}}test-pass-options{order=sideways}
// BADENUM-NEXT: ^
// BADENUM-NEXT: test-pass-options: option 'order': expected top-down|bottom-up

// MALFORMED: {{^}}test-pass-options{number=3
// MALFORMED-NEXT: ^
// MALFORMED-NEXT: "}"

// NOPASS:      {{^}}not-a-pass{x=1}
// NOPASS-NEXT: ^
// NOPASS-NEXT: 'not-a-pass' does not refer to a registered pass.
// NOPASS-NEXT: Currently registered passes are:
// NOPASS:        - test-pass-options{{$}}

// DUPLICATE: {{^}}test-pass-options{number=1 number=2}
// DUPLICATE-NEXT: ^
// DUPLICATE-NEXT: test-pass-options: duplicate option 'number'

// RUN: scair-opt %s -p 'test-pass-options{nubmer=3}' 2>&1 | filecheck %s --check-prefix=UNKNOWN
// RUN: scair-opt %s -p 'test-pass-options{number=three}' 2>&1 | filecheck %s --check-prefix=BADVALUE
// RUN: scair-opt %s -p 'test-pass-options{order=sideways}' 2>&1 | filecheck %s --check-prefix=BADENUM
// RUN: scair-opt %s -p 'test-pass-options{number=3' 2>&1 | filecheck %s --check-prefix=MALFORMED
// RUN: scair-opt %s -p 'not-a-pass{x=1}' 2>&1 | filecheck %s --check-prefix=NOPASS

builtin.module {}

// UNKNOWN: error: test-pass-options: unknown option 'nubmer', expected one of: number, ratio, label, flag, maybe, sizes, order

// BADVALUE: error: test-pass-options: option 'number': expected Int, got 'three'

// BADENUM: error: test-pass-options: option 'order': expected top-down|bottom-up, got 'sideways'

// MALFORMED: error: invalid pass pipeline 'test-pass-options{number=3': Expected "}":1:27, found ""

// NOPASS:      error: 'not-a-pass' does not refer to a registered pass.
// NOPASS-NEXT: Currently registered passes are:
// NOPASS:        - test-pass-options{number: Int = 1, ratio: Double = 0.5, label: String = default, flag: Boolean = false, maybe: Option[Int], sizes: Seq[Int], order: top-down|bottom-up = top-down}

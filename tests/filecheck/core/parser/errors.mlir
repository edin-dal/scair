// RUN: scair-opt %s --parsing-diagnostics --split-input-file | filecheck %s -DSOURCE_FILE=%s

%1, %2, %3 = "test.op"() : () -> (i8, i16)

// CHECK:       Parse error at [[SOURCE_FILE]]:3:42:

// CHECK:       %1, %2, %3 = "test.op"() : () -> (i8, i16)
// CHECK-NEXT:                                           ^
// CHECK-NEXT:  Number of results (3) does not match the number of the corresponding result types (2).

// -----

%1 = "test.op"() : () -> (i8)
"test.op"(%1) : () -> ()

// CHECK:       [[SOURCE_FILE]]:14:18:

// CHECK:       "test.op"(%1) : () -> ()
// CHECK-NEXT:                   ^
// CHECK-NEXT:  (BuiltinType | DialectType | TypeAlias)

// -----

%1 = "test.op"() : () -> (i8, i16)

// CHECK:       [[SOURCE_FILE]]:24:29:

// CHECK:       %1 = "test.op"() : () -> (i8, i16)
// CHECK-NEXT:                              ^
// CHECK-NEXT:  Number of results (1) does not match the number of the corresponding result types.

// -----

"unregistered.generic"() : () -> ()

// CHECK:       Parse error at [[SOURCE_FILE]]:34:23:

// CHECK:       "unregistered.generic"() : () -> ()
// CHECK-NEXT:                        ^
// CHECK-NEXT:  Operation unregistered.generic is not registered. If this is intended, use `--allow-unregistered-dialect`.

// -----

unregistered.custom

// CHECK:       Parse error at [[SOURCE_FILE]]:44:20:

// CHECK:       unregistered.custom
// CHECK-NEXT:                     ^
// CHECK-NEXT:  Operation unregistered.custom is not defined in any supported Dialect.

// -----

"builtin.module"() ({
    ^bb0:
    ^bb0:
})

// CHECK:       Parse error at [[SOURCE_FILE]]:56:9:

// CHECK:           ^bb0:
// CHECK-NEXT:          ^
// CHECK-NEXT:  Block cannot be defined twice within the same scope - ^bb0
package scair.dialects.example

import java.io.PrintWriter
import scair.scairdl.irdef._
import scair.clair.mirrored._
import java.io.File

object ExampleDialect
    extends ScaIRDLDialect(
      DialectDef(
        "example",
        Seq(OperationDef("example.op", "ExampleOp")),
        Seq(AttributeDef("example.attr", "ExampleAttr"))
      )
    )

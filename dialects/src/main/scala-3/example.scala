package scair.dialects.example

import java.io.PrintWriter
import scair.scairdl.irdef._
import org.w3c.dom.Attr
import java.io.File

object ExampleDialect {
  val dialect_def = DialectDef(
    "example",
    Seq(OperationDef("example.op", "ExampleOp")),
    Seq(AttributeDef("example.attr", "ExampleAttr"))
  )

  def main(args: Array[String]): Unit = {
    val code = dialect_def.print(0)
    val file = File(args(0))
    file.getParentFile().mkdirs();
    file.createNewFile();
    val writer = PrintWriter(file)
    writer.write(code)
    writer.flush()
    writer.close()
  }
}

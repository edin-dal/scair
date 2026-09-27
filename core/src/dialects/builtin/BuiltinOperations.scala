package scair.dialects.builtin

import fastparse.*
import scair.clair.*
import scair.ir.*
import scair.parse.*
import scair.print.Printer
import scair.utils.*

// ==------== //
//  ModuleOp  //
// ==------== //

given OperationCustomParser[ModuleOp]:

  // ==--- Custom Parsing ---== //
  def parse[$: P](
      resNames: Seq[String]
  )(using Parser): P[ModuleOp] =
    P(
      regionP()
    ).map(ModuleOp.apply)

  // ==----------------------== //

case class ModuleOp(
    body: Region
) extends DerivedOperation["builtin.module"]
    with SymbolTable derives OpDefs:

  override def customPrint(
      p: Printer
  ) =
    p.print("builtin.module ", regions(0))

case class UnrealizedConversionCastOp(
    inputs: Seq[Value[Attribute]] = Seq(),
    outputs: Seq[Result[Attribute]] = Seq(),
) extends DerivedOperation["builtin.unrealized_conversion_cast"] derives OpDefs

val BuiltinDialect =
  summonDialect[EmptyTuple, (ModuleOp, UnrealizedConversionCastOp)]
